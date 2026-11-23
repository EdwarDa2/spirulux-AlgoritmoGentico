import copy

import pytest

from src.modelo.parametros import (
    REQUERIDOS,
    cargar_escenario,
    cargar_json,
    k_extincion,
    valor,
    validar_escenario,
)


@pytest.fixture
def escenario():
    return cargar_escenario("escenario_base")


def test_el_escenario_base_carga_y_es_valido(escenario):
    assert escenario["nombre"] == "escenario_base"


def test_estan_todos_los_parametros_requeridos(escenario):
    for seccion, nombres in REQUERIDOS.items():
        for nombre in nombres:
            assert nombre in escenario[seccion]


def test_k_extincion_valor_conocido_farges():
    # Farges 2009 (LED 626 nm): Ea=225, Es=609, b=0.033  ->  k ≈ 244.3
    assert k_extincion(225.0, 609.0, 0.033) == pytest.approx(244.3, abs=0.1)


def test_k_extincion_sin_dispersion_es_igual_a_ea():
    assert k_extincion(225.0, 600.0, 0.0) == pytest.approx(225.0)


def test_k_guardado_coincide_con_la_formula(escenario):
    k = k_extincion(valor(escenario, "luz", "Ea"),
                    valor(escenario, "luz", "Es"),
                    valor(escenario, "luz", "b"))
    assert valor(escenario, "luz", "k_extincion") == pytest.approx(k, rel=0.01)


def test_rendimiento_energetico_maximo_no_supera_uno(escenario):
    assert 0 < valor(escenario, "crecimiento", "rho_M") <= 1


def test_fraccion_de_carbono_es_razonable(escenario):
    assert 0.4 < valor(escenario, "carbono", "fraccion_carbono_biomasa") < 0.6


def test_fraccion_de_carbono_coincide_con_masa_c_molar(escenario):
    # 12.011 g/mol de C dividido entre la masa C-molar de la biomasa
    esperado = 12.011e-3 / valor(escenario, "carbono", "masa_c_molar_biomasa")
    assert valor(escenario, "carbono", "fraccion_carbono_biomasa") == pytest.approx(esperado, rel=0.01)


def test_la_compensacion_es_menor_que_la_inhibicion(escenario):
    assert valor(escenario, "luz", "i_comp") < valor(escenario, "luz", "i_inhibicion")


def test_falta_un_parametro_lanza_error(escenario):
    roto = copy.deepcopy(escenario)
    del roto["crecimiento"]["K"]
    with pytest.raises(ValueError, match="crecimiento.K"):
        validar_escenario(roto)


def test_parametro_sin_fuente_lanza_error(escenario):
    roto = copy.deepcopy(escenario)
    roto["luz"]["Ea"]["fuente"] = ""
    with pytest.raises(ValueError, match="fuente"):
        validar_escenario(roto)


def test_estado_invalido_lanza_error(escenario):
    roto = copy.deepcopy(escenario)
    roto["luz"]["Ea"]["estado"] = "inventado"
    with pytest.raises(ValueError, match="estado"):
        validar_escenario(roto)


def test_valor_no_positivo_lanza_error(escenario):
    roto = copy.deepcopy(escenario)
    roto["crecimiento"]["K"]["valor"] = -5.0
    with pytest.raises(ValueError, match="positivo"):
        validar_escenario(roto)


def test_k_inconsistente_lanza_error(escenario):
    roto = copy.deepcopy(escenario)
    roto["luz"]["k_extincion"]["valor"] = 999.0
    with pytest.raises(ValueError, match="k_extincion"):
        validar_escenario(roto)


def test_referencia_tiene_los_casos_de_validacion():
    ref = cargar_json("referencia")
    assert ref["farges_2009_continuo"]["biomasa_salida"]["valor"] == pytest.approx(0.7)
    assert ref["reactor_casero_10L"]["productividad"]["valor"] == pytest.approx(0.12)
    # Coherencia interna: biomasa de salida ≈ productividad × tiempo de residencia
    f = ref["farges_2009_continuo"]
    assert f["productividad"]["valor"] * f["tiempo_residencia"]["valor"] == pytest.approx(
        f["biomasa_salida"]["valor"], rel=0.05)
    

def test_eficacia_led_dentro_de_los_limites_fisicos(escenario):
    # Mínimo DLC ~1.9 y límite teórico de lámparas con LED ~4.1 µmol/J
    assert 1.0 <= valor(escenario, "energia", "eficacia_led") <= 4.1


def test_acople_optico_entre_cero_y_uno(escenario):
    assert 0 < valor(escenario, "energia", "acople_optico") <= 1.0