"""Atenuación de luz dentro del cultivo (ley de Lambert-Beer).

Unidades:
    i0          intensidad incidente         [µmol fotones m⁻² s⁻¹]
    biomasa_gL  concentración de biomasa     [g L⁻¹]  (= kg m⁻³)
    espesor_cm  camino óptico del cultivo    [cm]
    k           coeficiente de extinción     [m² kg⁻¹]
"""
import numpy as np


def _validar_no_negativo(**valores):
    for nombre, valor in valores.items():
        if np.any(np.asarray(valor, dtype=float) < 0):
            raise ValueError(f"{nombre} no puede ser negativo")


def profundidad_optica(biomasa_gL, espesor_cm, k):
    """Profundidad óptica adimensional: tau = k * X * L."""
    _validar_no_negativo(biomasa_gL=biomasa_gL, espesor_cm=espesor_cm, k=k)
    espesor_m = np.asarray(espesor_cm, dtype=float) / 100.0
    return k * np.asarray(biomasa_gL, dtype=float) * espesor_m


def luz_promedio(i0, biomasa_gL, espesor_cm, k):
    """Intensidad promedio a lo largo del espesor del cultivo.

    I_prom = i0 * (1 - exp(-tau)) / tau, con tau = k * X * L.
    Admite números o arreglos de NumPy (se calcula de forma vectorizada).
    """
    _validar_no_negativo(i0=i0)
    tau = profundidad_optica(biomasa_gL, espesor_cm, k)
    es_pequeno = tau < 1e-9                       # evita 0/0 cuando tau -> 0
    tau_seguro = np.where(es_pequeno, 1.0, tau)
    factor = np.where(es_pequeno, 1.0 - tau / 2.0, -np.expm1(-tau_seguro) / tau_seguro)
    return np.asarray(i0, dtype=float) * factor


def perfil_luz(i0, biomasa_gL, espesor_cm, k, n_puntos=50):
    """Intensidad de luz en función de la profundidad: I(z) = i0 * exp(-k X z).

    Devuelve (z_cm, intensidad), ambos arreglos de largo n_puntos.
    """
    _validar_no_negativo(i0=i0)
    _validar_no_negativo(biomasa_gL=biomasa_gL, espesor_cm=espesor_cm, k=k)
    z_cm = np.linspace(0.0, espesor_cm, n_puntos)
    intensidad = i0 * np.exp(-k * biomasa_gL * (z_cm / 100.0))
    return z_cm, intensidad


def fraccion_zona_oscura(i0, biomasa_gL, espesor_cm, k, i_comp):
    """Fracción del espesor donde la luz cae por debajo de i_comp (zona oscura).

    i_comp es la intensidad de compensación: abajo de ella la célula gasta más
    de lo que fotosintetiza. Devuelve un valor entre 0 y 1.
    """
    _validar_no_negativo(i0=i0, biomasa_gL=biomasa_gL, espesor_cm=espesor_cm,
                         k=k, i_comp=i_comp)
    if i_comp <= 0:
        return 0.0
    if i0 <= i_comp:
        return 1.0
    atenuacion = k * biomasa_gL                   # [m⁻¹]
    if atenuacion == 0 or espesor_cm == 0:
        return 0.0
    z_comp_m = np.log(i0 / i_comp) / atenuacion   # profundidad donde I = i_comp
    fraccion_iluminada = z_comp_m / (espesor_cm / 100.0)
    return float(np.clip(1.0 - fraccion_iluminada, 0.0, 1.0))