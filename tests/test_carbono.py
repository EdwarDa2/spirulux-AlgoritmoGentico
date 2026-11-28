import numpy as np
import pytest

from src.modelo.carbono import (
    PCO2_ATMOSFERA_PA,
    alcalinidad,
    co2_disuelto,
    factor_carbono,
    fracciones_dic,
    pco2_equilibrio_pa,
    ph_desde_dic,
)
from src.modelo.parametros import cargar_escenario, valor


@pytest.fixture
def esc():
    return cargar_escenario("escenario_base")


@pytest.fixture
def medio(esc):
    """DIC y alcalinidad del medio SOT modificado de Kishi et al. 2020."""
    return valor(esc, "carbono", "dic_medio"), valor(esc, "carbono", "alcalinidad_medio")


# ---------- fracciones ----------

def test_las_fracciones_suman_uno(esc):
    for ph in (4.0, 6.38, 8.0, 9.8, 11.0, 13.0):
        assert sum(fracciones_dic(ph, esc)) == pytest.approx(1.0)


def test_en_pK1_hay_igual_CO2_que_bicarbonato(esc):
    f_co2, f_hco3, _ = fracciones_dic(valor(esc, "carbono", "pK1"), esc)
    assert f_co2 == pytest.approx(f_hco3, rel=0.01)


def test_en_pK2_hay_igual_bicarbonato_que_carbonato(esc):
    _, f_hco3, f_co3 = fracciones_dic(valor(esc, "carbono", "pK2"), esc)
    assert f_hco3 == pytest.approx(f_co3, rel=0.01)


def test_a_pH_bajo_domina_el_CO2_y_a_pH_alto_el_carbonato(esc):
    assert fracciones_dic(4.0, esc)[0] > 0.99
    assert fracciones_dic(12.0, esc)[2] > 0.99


# ---------- pH desde DIC y alcalinidad ----------

def test_ida_y_vuelta_ph_alcalinidad_ph(esc):
    for ph in (7.5, 8.5, 9.5, 10.5, 11.5):
        for dic in (0.05, 0.27, 0.45):
            alc = alcalinidad(dic, ph, esc)
            assert ph_desde_dic(dic, alc, esc) == pytest.approx(ph, abs=1e-6)


def test_medio_fresco_tiene_ph_cercano_a_10_5(esc, medio):
    # Kishi et al. 2020 ajustaron el medio a pH 10.5 con NaOH
    assert ph_desde_dic(*medio, esc) == pytest.approx(10.5, abs=0.15)


def test_inyectar_CO2_baja_el_ph_y_consumir_carbono_lo_sube(esc, medio):
    dic0, alc = medio
    base = ph_desde_dic(dic0, alc, esc)
    assert ph_desde_dic(dic0 + 0.05, alc, esc) < base
    assert ph_desde_dic(dic0 - 0.05, alc, esc) > base


def test_reproduce_el_rango_de_buffer_reportado_por_kishi(esc, medio):
    # Kishi: con ΔDIC entre 0 y 0.17 mol/L el pH va de ~10.4 a ~8.9
    dic0, alc = medio
    assert ph_desde_dic(dic0, alc, esc) == pytest.approx(10.4, abs=0.1)
    assert ph_desde_dic(dic0 + 0.17, alc, esc) == pytest.approx(8.9, abs=0.1)


def test_el_buffer_se_agota_con_mucho_CO2(esc, medio):
    dic0, alc = medio
    assert ph_desde_dic(dic0 + 0.25, alc, esc) < 8.0


def test_ph_vectorizado(esc, medio):
    dic0, alc = medio
    ph = ph_desde_dic(dic0 + np.array([0.0, 0.05, 0.10]), alc, esc)
    assert ph.shape == (3,) and np.all(np.diff(ph) < 0)


# ---------- CO2 de equilibrio ----------

def test_CO2_disuelto_es_una_fraccion_del_DIC(esc, medio):
    dic0, alc = medio
    ph = ph_desde_dic(dic0, alc, esc)
    assert 0 < co2_disuelto(dic0, ph, esc) < dic0


def test_el_CO2_de_equilibrio_supera_a_la_atmosfera_al_pasar_de_cero_a_0_02(esc, medio):
    # Kishi: con ΔDIC < 0.02 mol/L el CO2 de equilibrio ya excede el atmosférico (0.04 %)
    dic0, alc = medio
    p0 = pco2_equilibrio_pa(dic0, ph_desde_dic(dic0, alc, esc), esc)
    p1 = pco2_equilibrio_pa(dic0 + 0.02, ph_desde_dic(dic0 + 0.02, alc, esc), esc)
    assert p0 < PCO2_ATMOSFERA_PA < p1


def test_el_CO2_de_equilibrio_crece_con_el_DIC(esc, medio):
    dic0, alc = medio
    p = [pco2_equilibrio_pa(dic0 + d, ph_desde_dic(dic0 + d, alc, esc), esc) for d in (0.0, 0.05, 0.1, 0.17)]
    assert all(b > a for a, b in zip(p, p[1:]))


# ---------- limitación por carbono ----------

def test_sin_carbono_no_hay_crecimiento(esc):
    assert factor_carbono(0.0, esc) == 0.0


def test_a_K_TIC_el_factor_es_un_medio(esc):
    assert factor_carbono(valor(esc, "carbono", "K_TIC"), esc) == pytest.approx(0.5)


def test_con_el_DIC_del_medio_casi_no_hay_limitacion(esc, medio):
    assert factor_carbono(medio[0], esc) > 0.99


def test_el_factor_crece_con_el_DIC_y_esta_entre_0_y_1(esc):
    f = factor_carbono(np.array([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]), esc)
    assert np.all(np.diff(f) > 0) and np.all((f >= 0) & (f <= 1))


# ---------- validaciones ----------

def test_valores_negativos_lanzan_error(esc):
    with pytest.raises(ValueError):
        ph_desde_dic(-0.1, 0.4, esc)
    with pytest.raises(ValueError):
        factor_carbono(-1e-3, esc)
    with pytest.raises(ValueError):
        alcalinidad(-0.1, 9.0, esc)