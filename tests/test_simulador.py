import numpy as np
import pytest

from src.modelo.energia import potencia_led
from src.modelo.parametros import cargar_escenario, cargar_json, valor
from src.modelo.simulador import (
    NOMBRES_GENES,
    OBJETIVOS,
    ConfigSimulacion,
    desempaquetar_genes,
    simular,
)


@pytest.fixture
def esc():
    return cargar_escenario("escenario_base")

def genes_farges(luz=135.0, espesor=4.0, dilucion=24 / 62.5):
    """Caso continuo de Farges 2009: luz constante, 4 cm, residencia de 62.5 h."""
    return np.array([luz] * 8 + [espesor, dilucion, 0.0, 0.0])


# ---------- estructura del contrato ----------

def test_hay_12_genes_y_4_objetivos():
    assert len(NOMBRES_GENES) == 12
    assert NOMBRES_GENES[:2] == ("luz_bloque_1", "luz_bloque_2")
    assert NOMBRES_GENES[-4:] == ("espesor_cultivo", "tasa_dilucion", "inyeccion_co2", "potencia_mezclado")
    assert OBJETIVOS == ("productividad_biomasa", "energia_por_gramo", "eficiencia_co2", "estres_luminico")


def test_desempaquetar_separa_los_genes_por_nombre():
    g = desempaquetar_genes(np.arange(1, 13, dtype=float))
    assert g["luces"].shape == (1, 8)
    assert g["espesor_cm"][0] == 9 and g["dilucion_d"][0] == 10
    assert g["co2"][0] == 11 and g["mezclado"][0] == 12


def test_cantidad_de_genes_incorrecta_lanza_error():
    with pytest.raises(ValueError, match="genes"):
        desempaquetar_genes(np.ones(11))


def test_genes_negativos_lanzan_error():
    g = genes_farges()
    g[0] = -1.0
    with pytest.raises(ValueError, match="negativos"):
        desempaquetar_genes(g)


def test_espesor_cero_lanza_error():
    with pytest.raises(ValueError, match="espesor"):
        desempaquetar_genes(genes_farges(espesor=0.0))


def test_genes_no_finitos_lanzan_error():
    g = genes_farges()
    g[3] = np.nan
    with pytest.raises(ValueError, match="finitos"):
        desempaquetar_genes(g)


def test_config_valida_el_paso_de_tiempo_y_los_limites():
    with pytest.raises(ValueError, match="dt_h"):
        ConfigSimulacion(dt_h=0.7)
    with pytest.raises(ValueError, match="dias_eval"):
        ConfigSimulacion(dias=2, dias_eval=3)
    with pytest.raises(ValueError, match="X_min"):
        ConfigSimulacion(X_min=3.0, X_max=2.5)
    with pytest.raises(ValueError, match="base_estres"):
        ConfigSimulacion(base_estres="otra")
        


CFG_FARGES = ConfigSimulacion(dias=40, T_C=36.0)


# ---------- salidas ----------

def test_salidas_tienen_un_valor_por_individuo(esc):
    rng = np.random.default_rng(0)
    G = np.column_stack([rng.uniform(0, 300, (5, 8)), rng.uniform(2, 8, 5),
                         rng.uniform(0.1, 0.5, 5), rng.uniform(0, 5, 5), rng.uniform(0, 50, 5)])
    r = simular(G, esc)
    for nombre in OBJETIVOS + ("violacion", "biomasa_final_gL", "energia_led_kwh"):
        assert getattr(r, nombre).shape == (5,)
    assert r.a_matriz().shape == (5, 4)


def test_un_solo_individuo_funciona_con_vector_1d(esc):
    r = simular(genes_farges(), esc)
    assert r.productividad_biomasa.shape == (1,)


def test_eficiencia_co2_es_nan_y_lo_demas_es_finito(esc):
    r = simular(genes_farges(), esc)
    assert np.all(np.isnan(r.eficiencia_co2))
    for nombre in ("productividad_biomasa", "energia_por_gramo", "estres_luminico", "violacion"):
        assert np.all(np.isfinite(getattr(r, nombre)))
    assert any("carbono" in a for a in r.advertencias)
    assert any("mezclado" in a for a in r.advertencias)


def test_es_determinista(esc):
    a = simular(genes_farges(), esc)
    b = simular(genes_farges(), esc)
    assert np.array_equal(a.productividad_biomasa, b.productividad_biomasa)


def test_simular_en_poblacion_equivale_a_simular_uno_por_uno(esc):
    rng = np.random.default_rng(3)
    G = np.column_stack([rng.uniform(0, 300, (4, 8)), rng.uniform(2, 8, 4),
                         rng.uniform(0.1, 0.5, 4), np.zeros(4), np.zeros(4)])
    juntos = simular(G, esc)
    for i in range(4):
        solo = simular(G[i], esc)
        assert juntos.productividad_biomasa[i] == pytest.approx(solo.productividad_biomasa[0])
        assert juntos.energia_por_gramo[i] == pytest.approx(solo.energia_por_gramo[0])


# ---------- física y validación ----------

def test_reproduce_el_estado_estacionario_de_farges(esc):
    ref = cargar_json("referencia")["farges_2009_continuo"]
    r = simular(genes_farges(), esc, CFG_FARGES)
    # La biomasa llegó a un estado estacionario: crecimiento = dilución
    from src.modelo.crecimiento import tasa_crecimiento
    X = r.biomasa_final_gL[0]
    crecimiento = float(tasa_crecimiento(135.0, X, 4.0, 36.0, esc))
    assert crecimiento == pytest.approx(24 / 62.5 / 24 * X, rel=0.01)
    # Y queda dentro de la tolerancia usada contra el dato publicado (modelo ~15 % conservador)
    assert X == pytest.approx(ref["biomasa_salida"]["valor"], rel=0.20)
    assert r.productividad_biomasa[0] == pytest.approx(ref["productividad"]["valor"] * 24, rel=0.20)
    assert r.violacion[0] == 0.0


def test_el_resultado_converge_al_reducir_el_paso_de_tiempo(esc):
    a = simular(genes_farges(), esc, ConfigSimulacion(dias=40, T_C=36.0, dt_h=0.5))
    b = simular(genes_farges(), esc, ConfigSimulacion(dias=40, T_C=36.0, dt_h=0.25))
    assert a.productividad_biomasa[0] == pytest.approx(b.productividad_biomasa[0], rel=0.01)


def test_cultivo_apagado_se_lava_y_es_infactible(esc):
    g = genes_farges(luz=0.0)
    r = simular(g, esc)
    assert r.productividad_biomasa[0] < 0.02
    assert r.energia_led_kwh[0] == 0.0
    assert r.violacion[0] > 0          # la biomasa cae bajo X_min


def test_energia_led_coincide_con_potencia_por_tiempo(esc):
    cfg = ConfigSimulacion()
    r = simular(genes_farges(luz=135.0), esc, cfg)
    esperado = potencia_led(135.0, cfg.area_m2, esc) * 24 * cfg.dias_eval / 1000.0
    assert r.energia_led_kwh[0] == pytest.approx(esperado)


def test_energia_por_gramo_es_energia_entre_biomasa_cosechada(esc):
    cfg = ConfigSimulacion()
    r = simular(genes_farges(), esc, cfg)
    assert r.energia_por_gramo[0] == pytest.approx(
        r.energia_led_kwh[0] / (r.biomasa_cosechada_g[0] + cfg.delta_g))


def test_energia_por_gramo_no_depende_del_area_del_panel(esc):
    # El volumen crece igual que el área, así que el área se cancela
    a = simular(genes_farges(), esc, ConfigSimulacion(area_m2=0.1))
    b = simular(genes_farges(), esc, ConfigSimulacion(area_m2=1.0))
    assert a.energia_por_gramo[0] == pytest.approx(b.energia_por_gramo[0], rel=1e-6)
    assert a.productividad_biomasa[0] == pytest.approx(b.productividad_biomasa[0], rel=1e-6)


def test_biomasa_cosechada_coincide_con_productividad_por_volumen(esc):
    cfg = ConfigSimulacion()
    r = simular(genes_farges(), esc, cfg)
    volumen_L = cfg.area_m2 * 0.04 * 1000.0
    assert r.biomasa_cosechada_g[0] == pytest.approx(
        r.productividad_biomasa[0] * cfg.dias_eval * volumen_L)


# ---------- estrés lumínico ----------

def test_luz_baja_no_causa_estres(esc):
    assert simular(genes_farges(luz=50.0), esc).estres_luminico[0] == 0.0


def test_luz_muy_alta_con_poca_biomasa_causa_estres(esc):
    # Arranque con poca biomasa: la luz que ve el cultivo supera el umbral
    cfg = ConfigSimulacion(dias=2, dias_eval=2, biomasa_inicial=0.05)
    r = simular(genes_farges(luz=1500.0, espesor=2.0, dilucion=0.5), esc, cfg)
    assert r.estres_luminico[0] > 0


def test_un_cultivo_denso_se_autoprotege_pero_el_estres_incidente_no_lo_ve(esc):
    # En estado estacionario el cultivo es denso y la luz promedio baja del umbral.
    # El estrés sobre luz incidente, en cambio, sigue viendo los 1500 µmol.
    g = genes_farges(luz=1500.0, espesor=2.0, dilucion=0.5)
    assert simular(g, esc).estres_luminico[0] == 0.0
    assert simular(g, esc, ConfigSimulacion(base_estres="incidente")).estres_luminico[0] > 0


def test_el_estres_sobre_luz_incidente_nunca_es_menor_que_sobre_luz_promedio(esc):
    g = genes_farges(luz=400.0, espesor=6.0, dilucion=0.3)
    prom = simular(g, esc, ConfigSimulacion(base_estres="promedio"))
    inc = simular(g, esc, ConfigSimulacion(base_estres="incidente"))
    assert inc.estres_luminico[0] >= prom.estres_luminico[0]


# ---------- restricciones ----------

def test_biomasa_sobre_el_maximo_es_infactible(esc):
    # Dilución casi nula y luz alta: la biomasa se acumula por encima de X_max
    r = simular(genes_farges(luz=400.0, espesor=10.0, dilucion=0.01), esc,
                ConfigSimulacion(dias=20, X_max=0.5))
    assert r.violacion[0] > 0


# ---------- estado actual de las piezas pendientes ----------

def test_por_ahora_co2_y_mezclado_no_cambian_los_resultados(esc):
    # Documenta una limitación: al agregar carbono.py y el mezclado este test debe
    # actualizarse (y la advertencia correspondiente desaparecer).
    a = genes_farges()
    b = a.copy()
    b[10], b[11] = 4.0, 80.0
    ra, rb = simular(a, esc), simular(b, esc)
    assert ra.productividad_biomasa[0] == rb.productividad_biomasa[0]
    assert ra.energia_por_gramo[0] == rb.energia_por_gramo[0]


# ---------- series para las gráficas ----------

def test_series_tienen_la_forma_esperada(esc):
    cfg = ConfigSimulacion()
    r = simular(np.vstack([genes_farges(), genes_farges(luz=200.0)]), esc, cfg, series=True)
    T = cfg.dias * 48
    assert r.series["tiempo_h"].shape == (T,)
    for clave in ("biomasa", "luz_incidente", "luz_promedio", "potencia_led_w"):
        assert r.series[clave].shape == (2, T)
    assert simular(genes_farges(), esc).series is None


def test_el_perfil_de_8_bloques_se_repite_cada_dia(esc):
    luces = [0, 0, 100, 200, 200, 100, 0, 0]
    g = np.array(luces + [4.0, 0.3, 0.0, 0.0], dtype=float)
    r = simular(g, esc, ConfigSimulacion(dias=3), series=True)
    q = r.series["luz_incidente"][0]
    assert np.array_equal(q[:48], q[48:96])       # día 1 = día 2
    # Con dt = 0.5 h cada bloque de 3 h ocupa 6 pasos
    assert np.all(q[0:12] == 0.0)                 # bloques 1 y 2 (00-06 h)
    assert np.all(q[12:18] == 100.0)              # bloque 3 (06-09 h)
    assert np.all(q[18:30] == 200.0)              # bloques 4 y 5 (09-15 h)
    assert np.all(q[30:36] == 100.0)              # bloque 6 (15-18 h)
    assert np.all(q[36:48] == 0.0)                # bloques 7 y 8 (18-24 h)