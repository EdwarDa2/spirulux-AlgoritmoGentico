"""Química del carbono inorgánico del medio y limitación del crecimiento por carbono.

Ecuaciones: Kishi et al. 2020, Appl. Sci. 10:228 (ec. 1-3 y 5).
DIC = [CO2] + [HCO3-] + [CO3^2-]. La alcalinidad del medio (iones que acompañan
al carbono, sobre todo Na+) se conserva al inyectar o consumir CO2:
    Alc = [HCO3-] + 2[CO3^2-] + [OH-] - [H+]
Dado DIC y Alc, el pH sale de resolver esa ecuación (monótona en el pH).

Unidades: DIC y alcalinidad en mol L⁻¹, presiones en Pa.
Límites: constantes a 25 °C (el cultivo opera a ~35 °C) y sin corrección por
fuerza iónica más allá de que pK2 es aparente para el medio SOT.
"""
import numpy as np

from .parametros import valor

KW = 1.0e-14                 # mol² L⁻² (Kishi 2020)
PCO2_ATMOSFERA_PA = 40.5     # 0.04 % de 101325 Pa (Kishi 2020: "ca. 0.04 %")
PH_MIN, PH_MAX = 2.0, 14.0   # intervalo de búsqueda del pH


def _no_negativo(**valores):
    for nombre, v in valores.items():
        if np.any(np.asarray(v, dtype=float) < 0):
            raise ValueError(f"{nombre} no puede ser negativo")


def _constantes(escenario):
    K1 = 10.0 ** (-valor(escenario, "carbono", "pK1"))
    K2 = 10.0 ** (-valor(escenario, "carbono", "pK2"))
    return K1, K2


def fracciones_dic(ph, escenario):
    """Fracciones (CO2, HCO3-, CO3^2-) del DIC a un pH dado. Suman 1."""
    K1, K2 = _constantes(escenario)
    h = 10.0 ** (-np.asarray(ph, dtype=float))
    d = 1.0 + K1 / h + K1 * K2 / h ** 2
    return 1.0 / d, (K1 / h) / d, (K1 * K2 / h ** 2) / d


def alcalinidad(dic, ph, escenario):
    """Alcalinidad [mol L⁻¹] que corresponde a un DIC y un pH."""
    _no_negativo(dic=dic)
    _, f_hco3, f_co3 = fracciones_dic(ph, escenario)
    h = 10.0 ** (-np.asarray(ph, dtype=float))
    return np.asarray(dic, dtype=float) * (f_hco3 + 2.0 * f_co3) + KW / h - h


def ph_desde_dic(dic, alcalinidad_mol_L, escenario):
    """pH del medio dado su DIC y su alcalinidad (bisección, vectorizada)."""
    _no_negativo(dic=dic, alcalinidad_mol_L=alcalinidad_mol_L)
    dic, alc = np.broadcast_arrays(np.asarray(dic, dtype=float),
                                   np.asarray(alcalinidad_mol_L, dtype=float))
    lo = np.full(dic.shape, PH_MIN)
    hi = np.full(dic.shape, PH_MAX)
    for _ in range(60):
        medio = 0.5 * (lo + hi)
        mayor = alcalinidad(dic, medio, escenario) > alc   # la alcalinidad crece con el pH
        hi = np.where(mayor, medio, hi)
        lo = np.where(mayor, lo, medio)
    return 0.5 * (lo + hi)


def co2_disuelto(dic, ph, escenario):
    """CO2 disuelto [mol L⁻¹]."""
    _no_negativo(dic=dic)
    f_co2, _, _ = fracciones_dic(ph, escenario)
    return np.asarray(dic, dtype=float) * f_co2


def pco2_equilibrio_pa(dic, ph, escenario):
    """Presión parcial de CO2 [Pa] en equilibrio con el medio (ley de Henry)."""
    return co2_disuelto(dic, ph, escenario) / valor(escenario, "carbono", "Hcp")


def factor_carbono(dic, escenario):
    """Factor de limitación por carbono (Monod sobre el carbono inorgánico total), entre 0 y 1."""
    _no_negativo(dic=dic)
    dic = np.asarray(dic, dtype=float)
    return dic / (valor(escenario, "carbono", "K_TIC") + dic)