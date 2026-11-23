import copy

import numpy as np
import pytest

from src.modelo.energia import energia_led_kwh, energia_led_perfil_kwh, potencia_led
from src.modelo.parametros import cargar_escenario, cargar_json, valor


@pytest.fixture
def esc():
    return cargar_escenario("escenario_base")


def test_sin_luz_no_hay_consumo(esc):
    assert potencia_led(0.0, 0.1, esc) == 0.0


def test_sin_area_no_hay_consumo(esc):
    assert potencia_led(135.0, 0.0, esc) == 0.0


def test_valor_conocido(esc):
    # 100 µmol m⁻² s⁻¹ sobre 1 m² son 100 µmol/s; dividido entre Γ*η da la potencia
    gamma = valor(esc, "energia", "eficacia_led")
    acople = valor(esc, "energia", "acople_optico")
    assert potencia_led(100.0, 1.0, esc) == pytest.approx(100.0 / (gamma * acople))


def test_la_potencia_es_lineal_en_luz_y_en_area(esc):
    base = potencia_led(100.0, 0.1, esc)
    assert potencia_led(200.0, 0.1, esc) == pytest.approx(2 * base)
    assert potencia_led(100.0, 0.2, esc) == pytest.approx(2 * base)


def test_un_led_menos_eficaz_consume_mas(esc):
    peor = copy.deepcopy(esc)
    peor["energia"]["eficacia_led"]["valor"] = valor(esc, "energia", "eficacia_led") / 2
    assert potencia_led(100.0, 0.1, peor) == pytest.approx(2 * potencia_led(100.0, 0.1, esc))


def test_peor_acople_optico_consume_mas(esc):
    peor = copy.deepcopy(esc)
    peor["energia"]["acople_optico"]["valor"] = 0.5
    assert potencia_led(100.0, 0.1, peor) > potencia_led(100.0, 0.1, esc)


def test_energia_es_potencia_por_tiempo(esc):
    # Se elige q para que la potencia sea exactamente 36 W en 1 m²
    q = 36.0 * valor(esc, "energia", "eficacia_led") * valor(esc, "energia", "acople_optico")
    assert potencia_led(q, 1.0, esc) == pytest.approx(36.0)
    assert energia_led_kwh(q, 1.0, 24.0, esc) == pytest.approx(0.864)   # 36 W * 24 h


def test_perfil_por_bloques_coincide_con_la_suma_manual(esc):
    perfil = np.array([0.0, 0.0, 100.0, 200.0, 200.0, 100.0, 0.0, 0.0])
    esperado = sum(energia_led_kwh(q, 0.1, 3.0, esc) for q in perfil)
    assert energia_led_perfil_kwh(perfil, 0.1, 3.0, esc) == pytest.approx(esperado)


def test_perfil_apagado_no_consume(esc):
    assert energia_led_perfil_kwh(np.zeros(8), 0.1, 3.0, esc) == 0.0


def test_poblacion_devuelve_un_valor_por_individuo(esc):
    poblacion = np.array([[0.0] * 8, [100.0] * 8, [200.0] * 8])
    e = energia_led_perfil_kwh(poblacion, 0.1, 3.0, esc)
    assert e.shape == (3,)
    assert e[0] == 0.0 and e[2] == pytest.approx(2 * e[1])


def test_valores_negativos_lanzan_error(esc):
    with pytest.raises(ValueError):
        potencia_led(-1.0, 0.1, esc)
    with pytest.raises(ValueError):
        energia_led_kwh(100.0, 0.1, -2.0, esc)


def test_la_referencia_del_reactor_casero_es_coherente():
    # Datos publicados: 36 W * 24 h / 1.34 g/d = 0.645 kWh/g  (verifica los datos, no el código)
    c = cargar_json("referencia")["reactor_casero_10L"]
    kwh_dia = c["potencia_led"]["valor"] * 24 / 1000
    assert kwh_dia / c["biomasa_diaria"]["valor"] == pytest.approx(c["consumo_especifico"]["valor"], rel=0.01)
    assert 1.0 / c["consumo_especifico"]["valor"] == pytest.approx(c["eficiencia_energetica"]["valor"], rel=0.01)