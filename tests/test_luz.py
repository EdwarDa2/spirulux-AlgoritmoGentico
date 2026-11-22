import numpy as np
import pytest

from src.modelo.luz import luz_promedio, profundidad_optica

# Con k=100 m²/kg, X=1 g/L y L=1 cm resulta tau = 100 * 1 * 0.01 = 1
K, X, L = 100.0, 1.0, 1.0


def test_profundidad_optica_valor_conocido():
    assert profundidad_optica(X, L, K) == pytest.approx(1.0)


def test_sin_biomasa_la_luz_no_se_atenua():
    assert luz_promedio(200.0, 0.0, 5.0, K) == pytest.approx(200.0)


def test_sin_espesor_la_luz_no_se_atenua():
    assert luz_promedio(200.0, 1.0, 0.0, K) == pytest.approx(200.0)


def test_valor_conocido_tau_uno():
    esperado = 200.0 * (1 - np.exp(-1.0)) / 1.0
    assert luz_promedio(200.0, X, L, K) == pytest.approx(esperado)


def test_continuidad_cerca_de_tau_cero():
    # No debe haber saltos ni NaN al pasar de la rama estable a la de serie
    a = luz_promedio(100.0, 1e-9, 1.0, K)
    b = luz_promedio(100.0, 1e-5, 1.0, K)
    assert np.isfinite(a) and np.isfinite(b)
    assert a == pytest.approx(100.0, rel=1e-6)
    assert b == pytest.approx(100.0, rel=1e-3)


def test_mas_biomasa_da_menos_luz():
    assert luz_promedio(200.0, 2.0, 5.0, K) < luz_promedio(200.0, 1.0, 5.0, K)


def test_mas_espesor_da_menos_luz():
    assert luz_promedio(200.0, 1.0, 10.0, K) < luz_promedio(200.0, 1.0, 5.0, K)


def test_luz_promedio_entre_cero_e_i0():
    i = luz_promedio(300.0, 1.5, 8.0, K)
    assert 0.0 < i <= 300.0


def test_vectorizado_con_arreglos():
    i0 = np.array([0.0, 100.0, 200.0])
    res = luz_promedio(i0, 1.0, 5.0, K)
    assert res.shape == (3,)
    assert res[0] == 0.0


def test_valores_negativos_lanzan_error():
    with pytest.raises(ValueError):
        luz_promedio(100.0, -1.0, 5.0, K)
    with pytest.raises(ValueError):
        luz_promedio(-5.0, 1.0, 5.0, K)