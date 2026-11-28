import numpy as np
import pytest

from src.modelo.simulador import (
    NOMBRES_GENES,
    OBJETIVOS,
    ConfigSimulacion,
    desempaquetar_genes,
)


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