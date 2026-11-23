"""Carga y validación de los parámetros del modelo (config/*.json).

Cada parámetro tiene: valor, unidad, fuente y estado
('literatura', 'derivado' o 'provisional').
"""
import json
from pathlib import Path

import numpy as np

RUTA_CONFIG = Path(__file__).resolve().parents[2] / "config"
ESTADOS_VALIDOS = {"literatura", "derivado", "provisional"}

REQUERIDOS = {
    "luz": ["Ea", "Es", "b", "k_extincion", "i_comp", "i_inhibicion"],
    "crecimiento": ["rho_M", "K", "phi", "T_opt", "fT_A", "fT_E1", "fT_B",
                    "fT_E2", "pH_optimo"],
    "carbono": ["masa_c_molar_biomasa", "fraccion_carbono_biomasa", "K_TIC"],
    "energia": ["eficacia_led", "conversion_par", "acople_optico"],
}


def k_extincion(Ea, Es, b):
    """Coeficiente de extinción de dos flujos: sqrt(Ea * (Ea + 2*b*Es)) [m² kg⁻¹]."""
    return float(np.sqrt(Ea * (Ea + 2.0 * b * Es)))


def validar_escenario(datos):
    """Revisa que el escenario esté completo y bien formado. Lanza ValueError si no."""
    for seccion, nombres in REQUERIDOS.items():
        if seccion not in datos:
            raise ValueError(f"Falta la sección '{seccion}'")
        for nombre in nombres:
            if nombre not in datos[seccion]:
                raise ValueError(f"Falta el parámetro '{seccion}.{nombre}'")
            p = datos[seccion][nombre]
            for campo in ("valor", "unidad", "fuente", "estado"):
                if campo not in p or p[campo] in (None, ""):
                    raise ValueError(f"'{seccion}.{nombre}' no tiene '{campo}'")
            if p["estado"] not in ESTADOS_VALIDOS:
                raise ValueError(f"'{seccion}.{nombre}' tiene estado inválido: {p['estado']}")
            if not isinstance(p["valor"], (int, float)) or p["valor"] <= 0:
                raise ValueError(f"'{seccion}.{nombre}' debe ser un número positivo")

    luz = datos["luz"]
    k_calculado = k_extincion(luz["Ea"]["valor"], luz["Es"]["valor"], luz["b"]["valor"])
    if abs(k_calculado - luz["k_extincion"]["valor"]) / k_calculado > 0.01:
        raise ValueError("k_extincion no coincide con sqrt(Ea*(Ea+2*b*Es))")
    return datos


def cargar_json(nombre):
    with open(RUTA_CONFIG / f"{nombre}.json", encoding="utf-8") as f:
        return json.load(f)


def cargar_escenario(nombre="escenario_base"):
    """Carga un escenario de parámetros y lo valida."""
    return validar_escenario(cargar_json(nombre))


def valor(datos, seccion, nombre):
    """Atajo para obtener solo el número de un parámetro."""
    return datos[seccion][nombre]["valor"]