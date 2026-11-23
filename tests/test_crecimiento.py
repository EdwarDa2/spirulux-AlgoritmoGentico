import numpy as np
import pytest

from src.modelo.crecimiento import factor_temperatura, tasa_crecimiento
from src.modelo.luz import luz_promedio
from src.modelo.parametros import cargar_escenario, valor


@pytest.fixture
def esc():
    return cargar_escenario("escenario_base")


# ---------- factor de temperatura ----------

def test_factor_temperatura_tiene_su_maximo_cerca_del_optimo(esc):
    T = np.linspace(15, 45, 3001)
    f = factor_temperatura(T, esc)
    assert T[f.argmax()] == pytest.approx(valor(esc, "crecimiento", "T_opt"), abs=0.5)
    assert f.max() == pytest.approx(1.0, abs=0.05)


def test_factor_temperatura_cae_lejos_del_optimo(esc):
    assert factor_temperatura(25.0, esc) < 0.7
    assert factor_temperatura(45.0, esc) < 0.5


def test_factor_temperatura_es_positivo_y_vectorizable(esc):
    f = factor_temperatura(np.array([20.0, 30.0, 40.0]), esc)
    assert f.shape == (3,) and np.all(f > 0)


# ---------- tasa de crecimiento ----------

def test_sin_biomasa_no_hay_crecimiento(esc):
    assert tasa_crecimiento(135.0, 0.0, 4.0, 35.0, esc) == 0.0


def test_sin_luz_no_hay_crecimiento(esc):
    assert tasa_crecimiento(0.0, 0.5, 4.0, 35.0, esc) == 0.0


def test_luz_bajo_el_punto_de_compensacion_no_hay_crecimiento(esc):
    i_comp = valor(esc, "luz", "i_comp")
    assert tasa_crecimiento(i_comp * 0.9, 0.5, 4.0, 35.0, esc) == 0.0


def test_sin_espesor_no_hay_crecimiento(esc):
    assert tasa_crecimiento(135.0, 0.5, 0.0, 35.0, esc) == 0.0


def test_mas_luz_da_mas_crecimiento(esc):
    r = [tasa_crecimiento(q, 0.5, 4.0, 35.0, esc) for q in (25, 50, 100, 200, 400)]
    assert all(b > a for a, b in zip(r, r[1:]))


def test_la_respuesta_a_la_luz_se_satura(esc):
    # Duplicar una luz ya alta debe dar mucho menos del doble
    r1 = tasa_crecimiento(1000.0, 0.5, 4.0, 35.0, esc)
    r2 = tasa_crecimiento(2000.0, 0.5, 4.0, 35.0, esc)
    assert r2 < 1.5 * r1


def test_con_poca_biomasa_mas_celulas_dan_mas_productividad(esc):
    assert tasa_crecimiento(135.0, 0.2, 4.0, 35.0, esc) > tasa_crecimiento(135.0, 0.1, 4.0, 35.0, esc)


def test_coincide_con_la_aproximacion_lineal_a_luz_baja(esc):
    # Con G << K el modelo se reduce a r ≈ rho_M*phi*Ea*X*luz_promedio (consistencia con luz.py)
    q, X, L_cm, T = 10.0, 0.05, 4.0, 35.0
    k = valor(esc, "luz", "k_extincion")
    esperado = (valor(esc, "crecimiento", "rho_M") * valor(esc, "crecimiento", "phi")
                * valor(esc, "luz", "Ea") * X * luz_promedio(q, X, L_cm, k)
                * 3600.0 * factor_temperatura(T, esc))
    assert tasa_crecimiento(q, X, L_cm, T, esc) == pytest.approx(esperado, rel=0.12)


def test_nunca_supera_la_cota_fisica_de_fotones(esc):
    # Aunque todos los fotones se absorbieran con rendimiento máximo:
    # r <= rho_M * phi * q / L   (g L⁻¹ h⁻¹)
    rho, phi = valor(esc, "crecimiento", "rho_M"), valor(esc, "crecimiento", "phi")
    for q in (10, 50, 135, 400, 1500):
        for X in (0.05, 0.3, 1.0, 3.0):
            for L_cm in (1, 4, 10):
                cota = rho * phi * q / (L_cm / 100.0) * 3600.0
                assert tasa_crecimiento(q, X, L_cm, 34.7, esc) <= cota * 1.0001


def test_f_carbono_escala_linealmente(esc):
    base = tasa_crecimiento(135.0, 0.5, 4.0, 35.0, esc)
    assert tasa_crecimiento(135.0, 0.5, 4.0, 35.0, esc, f_carbono=0.5) == pytest.approx(0.5 * base)


def test_vectorizado_con_poblacion(esc):
    q = np.array([50.0, 135.0, 300.0])
    X = np.array([0.3, 0.5, 0.8])
    L = np.array([2.0, 4.0, 8.0])
    r = tasa_crecimiento(q, X, L, 35.0, esc)
    assert r.shape == (3,) and np.all(r > 0)
    for i in range(3):   # coincide elemento a elemento con la versión escalar
        assert r[i] == pytest.approx(tasa_crecimiento(q[i], X[i], L[i], 35.0, esc))


def test_valores_negativos_lanzan_error(esc):
    with pytest.raises(ValueError):
        tasa_crecimiento(-1.0, 0.5, 4.0, 35.0, esc)
    with pytest.raises(ValueError):
        tasa_crecimiento(100.0, -0.5, 4.0, 35.0, esc)