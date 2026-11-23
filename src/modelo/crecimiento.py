"""Crecimiento de la biomasa de Arthrospira platensis (modelo de Cornet).

Fuentes: Farges et al. 2009 (ec. 12 y 16) y Poughon et al. 2021 (ec. 10 y 11).

Velocidad local:  r = rho_M * phi * Ea * X * K*G / (K + G)
con G(z) = q * exp(-k*X*z) (atenuación exponencial) y promedio sobre la zona
iluminada (G > i_comp). La zona oscura no crece ni respira (supuesto de Poughon).
Con esa atenuación la integral tiene forma cerrada, así que no hay bucles.

Unidades de entrada:
    q [µmol m⁻² s⁻¹]   biomasa_gL [g L⁻¹]   espesor_cm [cm]   T_C [°C]
Salida: g L⁻¹ h⁻¹
"""
import numpy as np

from .luz import _validar_no_negativo
from .parametros import valor

R_GAS = 8.314  # J mol⁻¹ K⁻¹


def factor_temperatura(T_C, escenario):
    """Factor de temperatura de Cornet (Poughon 2021, ec. 11). Máximo ≈ 1 cerca de T_opt."""
    T = np.asarray(T_C, dtype=float) + 273.15
    A = valor(escenario, "crecimiento", "fT_A")
    E1 = valor(escenario, "crecimiento", "fT_E1")
    B = valor(escenario, "crecimiento", "fT_B")
    E2 = valor(escenario, "crecimiento", "fT_E2")
    return A * np.exp(-E1 / (R_GAS * T)) / (1.0 + B * np.exp(-E2 / (R_GAS * T)))


def tasa_crecimiento(q, biomasa_gL, espesor_cm, T_C, escenario, f_carbono=1.0):
    """Productividad volumétrica de biomasa [g L⁻¹ h⁻¹].

    f_carbono: factor de limitación por carbono entre 0 y 1 (1 = sin limitación).
    Admite números o arreglos de NumPy (vectorizado).
    """
    _validar_no_negativo(q=q, biomasa_gL=biomasa_gL, espesor_cm=espesor_cm)
    k = valor(escenario, "luz", "k_extincion")
    Ea = valor(escenario, "luz", "Ea")
    i_comp = valor(escenario, "luz", "i_comp")
    rho_M = valor(escenario, "crecimiento", "rho_M")
    K = valor(escenario, "crecimiento", "K")
    phi = valor(escenario, "crecimiento", "phi")

    q = np.asarray(q, dtype=float)
    X = np.asarray(biomasa_gL, dtype=float)          # g/L = kg/m³
    L = np.asarray(espesor_cm, dtype=float) / 100.0  # m

    a = k * X                                        # atenuación [m⁻¹]
    hay_luz = (q > i_comp) & (L > 0)

    # Profundidad iluminada: donde G cae hasta i_comp
    q_seguro = np.where(hay_luz, q, i_comp * 2.0)
    a_pos = a > 1e-9
    a_seguro = np.where(a_pos, a, 1.0)
    L0 = np.where(a_pos, np.log(q_seguro / i_comp) / a_seguro, np.inf)
    Z = np.where(hay_luz, np.minimum(L, L0), 0.0)

    # ∫0^Z K*G/(K+G) dz con G = q*exp(-a z)  ->  (K/a) * ln((K+q)/(K+G(Z)))
    G_Z = q * np.exp(-a * Z)
    integral = np.where(
        a_pos,
        (K / a_seguro) * np.log((K + q) / (K + G_Z)),
        K * q / (K + q) * Z,                          # límite cuando a -> 0
    )
    L_seguro = np.where(L > 0, L, 1.0)
    r_kg_m3_s = rho_M * phi * Ea * X * integral / L_seguro
    r = r_kg_m3_s * 3600.0                            # kg m⁻³ h⁻¹ = g L⁻¹ h⁻¹
    return np.where(hay_luz, r, 0.0) * factor_temperatura(T_C, escenario) * f_carbono