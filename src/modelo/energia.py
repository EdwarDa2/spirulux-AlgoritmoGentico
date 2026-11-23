"""Energía del sistema de iluminación LED.

Potencia eléctrica necesaria para entregar una densidad de fotones q a un
área iluminada A:

    P_LED [W] = q [µmol m⁻² s⁻¹] * A [m²] / (Γ [µmol J⁻¹] * η_acople)

Γ es la eficacia fotónica del LED (fotones emitidos por joule eléctrico) y
η_acople la fracción de fotones emitidos que llega al cultivo (1 = ideal).

Este módulo solo cubre el LED. La energía del mezclado se agregará aparte
(pendiente de fuente) sin cambiar estas funciones.
"""
import numpy as np

from .luz import _validar_no_negativo
from .parametros import valor


def potencia_led(q, area_m2, escenario):
    """Potencia eléctrica del LED [W] para una densidad de fotones q [µmol m⁻² s⁻¹]."""
    _validar_no_negativo(q=q, area_m2=area_m2)
    eficacia = valor(escenario, "energia", "eficacia_led")
    acople = valor(escenario, "energia", "acople_optico")
    return np.asarray(q, dtype=float) * np.asarray(area_m2, dtype=float) / (eficacia * acople)


def energia_led_kwh(q, area_m2, horas, escenario):
    """Energía del LED [kWh] si q se mantiene constante durante 'horas'."""
    _validar_no_negativo(horas=horas)
    return potencia_led(q, area_m2, escenario) * np.asarray(horas, dtype=float) / 1000.0


def energia_led_perfil_kwh(q_por_bloque, area_m2, horas_por_bloque, escenario, eje=-1):
    """Energía total [kWh] de un perfil de luz por bloques (suma sobre el eje del tiempo).

    q_por_bloque puede ser un perfil (n_bloques,) o una población (N, n_bloques).
    """
    potencias = potencia_led(q_por_bloque, area_m2, escenario)
    _validar_no_negativo(horas_por_bloque=horas_por_bloque)
    return np.sum(potencias, axis=eje) * horas_por_bloque / 1000.0