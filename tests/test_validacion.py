"""Validación del modelo contra datos publicados (config/referencia.json)."""
import pytest

from src.modelo.crecimiento import tasa_crecimiento
from src.modelo.parametros import cargar_escenario, cargar_json


def _biomasa_estacionaria(q, L_cm, T_C, D_h, esc):
    """Resuelve r(X) = D*X por bisección (quimiostato en estado estacionario)."""
    lo, hi = 0.01, 5.0
    for _ in range(80):
        medio = (lo + hi) / 2
        if tasa_crecimiento(q, medio, L_cm, T_C, esc) - D_h * medio > 0:
            lo = medio
        else:
            hi = medio
    return (lo + hi) / 2


def test_reproduce_el_caso_continuo_de_farges_2009():
    esc = cargar_escenario("escenario_base")
    ref = cargar_json("referencia")["farges_2009_continuo"]
    D = 1.0 / ref["tiempo_residencia"]["valor"]
    X = _biomasa_estacionaria(ref["q_incidente"]["valor"], ref["espesor"]["valor"] * 100.0,
                              ref["temperatura"]["valor"], D, esc)
    productividad = float(tasa_crecimiento(ref["q_incidente"]["valor"], X,
                                           ref["espesor"]["valor"] * 100.0,
                                           ref["temperatura"]["valor"], esc))
    # Tolerancia de 20 %: el modelo usa una atenuación exponencial simple, no la solución
    # de dos flujos del artículo. Hoy queda unos 15 % por debajo (conservador).
    assert X == pytest.approx(ref["biomasa_salida"]["valor"], rel=0.20)
    assert productividad == pytest.approx(ref["productividad"]["valor"], rel=0.20)


def test_el_modelo_no_sobreestima_el_caso_de_farges():
    # Guarda de seguridad: si algún día el modelo sobreestima mucho, algo se rompió
    esc = cargar_escenario("escenario_base")
    ref = cargar_json("referencia")["farges_2009_continuo"]
    D = 1.0 / ref["tiempo_residencia"]["valor"]
    X = _biomasa_estacionaria(ref["q_incidente"]["valor"], ref["espesor"]["valor"] * 100.0,
                              ref["temperatura"]["valor"], D, esc)
    assert X < ref["biomasa_salida"]["valor"] * 1.10