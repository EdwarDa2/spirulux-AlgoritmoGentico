"""Simulador del fotobiorreactor: une luz, crecimiento y energía del LED.

CONTRATO CON EL AG (no cambiar sin avisar a la otra persona)
    entrada : genes con forma (N, 12) o (12,), en el orden de NOMBRES_GENES
    salida  : ResultadoSimulacion, con un valor por individuo para cada objetivo

Modelo: reactor de panel plano iluminado por una cara, bien mezclado, en
quimiostato (cosecha continua con dilución D). El perfil de 8 bloques de 3 h se
repite cada día. Los días y las horas son TIEMPO SIMULADO (pasos de dt_h horas).

    dX/dt = r(q(t), X, L, T) - D*X        [g L⁻¹ h⁻¹]   (Euler explícito)

Los objetivos se calculan sobre los últimos `dias_eval` días de la simulación.
"""
from dataclasses import dataclass, field

import numpy as np

from .crecimiento import tasa_crecimiento
from .energia import potencia_led
from .luz import luz_promedio
from .parametros import valor

N_BLOQUES = 8
HORAS_POR_BLOQUE = 3.0
NOMBRES_GENES = tuple(
    [f"luz_bloque_{i}" for i in range(1, N_BLOQUES + 1)]
    + ["espesor_cultivo", "tasa_dilucion", "inyeccion_co2", "potencia_mezclado"]
)
OBJETIVOS = ("productividad_biomasa", "energia_por_gramo", "eficiencia_co2", "estres_luminico")

ADVERTENCIAS = (
    "eficiencia_co2 no disponible (NaN): falta el modelo de carbono (carbono.py)",
    "inyeccion_co2 y potencia_mezclado todavia no afectan los resultados",
    "energia_por_gramo incluye solo la energia del LED (falta la del mezclado)",
)


@dataclass(frozen=True)
class ConfigSimulacion:
    """Entradas de la simulación (las de la propuesta que no son genes).

    X_max = 2.5 g/L: rango máximo usado por Poughon et al. 2021 (registro F2).
    El resto son decisiones de diseño provisionales, ajustables.
    """
    dias: int = 10                 # horizonte simulado
    dias_eval: int = 2             # últimos días sobre los que se miden los objetivos
    dt_h: float = 0.5              # paso de tiempo [h]; debe dividir 3 h
    T_C: float = 35.0              # temperatura del cultivo [°C]
    biomasa_inicial: float = 0.3   # [g/L]
    area_m2: float = 0.1           # área iluminada del panel [m²]
    X_min: float = 0.1             # biomasa mínima (evita el lavado) [g/L]
    X_max: float = 2.5             # biomasa máxima [g/L]
    base_estres: str = "promedio"  # 'promedio' (luz que ve el cultivo) o 'incidente'
    delta_g: float = 1e-6          # evita dividir entre cero en energía por gramo [g]

    def __post_init__(self):
        if self.dt_h <= 0 or abs(HORAS_POR_BLOQUE / self.dt_h - round(HORAS_POR_BLOQUE / self.dt_h)) > 1e-9:
            raise ValueError("dt_h debe ser positivo y dividir exactamente 3 horas")
        if self.dias < 1 or self.dias_eval < 1 or self.dias_eval > self.dias:
            raise ValueError("se requiere 1 <= dias_eval <= dias")
        if self.base_estres not in ("promedio", "incidente"):
            raise ValueError("base_estres debe ser 'promedio' o 'incidente'")
        if min(self.area_m2, self.biomasa_inicial, self.X_min, self.X_max, self.delta_g) <= 0:
            raise ValueError("area, biomasa inicial, limites y delta deben ser positivos")
        if self.X_min >= self.X_max:
            raise ValueError("X_min debe ser menor que X_max")


@dataclass
class ResultadoSimulacion:
    """Un valor por individuo (arreglos de largo N) en cada campo."""
    productividad_biomasa: np.ndarray   # g L⁻¹ d⁻¹   (maximizar)
    energia_por_gramo: np.ndarray       # kWh g⁻¹     (minimizar)
    eficiencia_co2: np.ndarray          # 0-1         (maximizar)  NaN por ahora
    estres_luminico: np.ndarray         # µmol m⁻²    (minimizar)
    violacion: np.ndarray               # 0 = factible; mayor = peor
    energia_led_kwh: np.ndarray
    energia_mezclado_kwh: np.ndarray
    biomasa_cosechada_g: np.ndarray
    biomasa_final_gL: np.ndarray
    advertencias: list = field(default_factory=lambda: list(ADVERTENCIAS))
    series: dict = None                 # solo si simular(..., series=True)

    def a_matriz(self):
        """Matriz (N, 4) con los objetivos en el orden de OBJETIVOS."""
        return np.column_stack([getattr(self, nombre) for nombre in OBJETIVOS])


def desempaquetar_genes(genes):
    """Valida los genes y los separa por nombre. Devuelve arreglos de largo N."""
    g = np.atleast_2d(np.asarray(genes, dtype=float))
    if g.ndim != 2 or g.shape[1] != len(NOMBRES_GENES):
        raise ValueError(f"se esperan {len(NOMBRES_GENES)} genes por individuo, llegaron {g.shape[-1]}")
    if not np.all(np.isfinite(g)):
        raise ValueError("los genes deben ser numeros finitos")
    if np.any(g < 0):
        raise ValueError("los genes no pueden ser negativos")
    if np.any(g[:, N_BLOQUES] <= 0):
        raise ValueError("espesor_cultivo debe ser mayor que cero")
    return {
        "luces": g[:, :N_BLOQUES],
        "espesor_cm": g[:, N_BLOQUES],
        "dilucion_d": g[:, N_BLOQUES + 1],
        "co2": g[:, N_BLOQUES + 2],
        "mezclado": g[:, N_BLOQUES + 3],
    }
    

def simular(genes, escenario, config=None, series=False):
    """Simula el cultivo de cada individuo y devuelve sus objetivos."""
    cfg = config or ConfigSimulacion()
    g = desempaquetar_genes(genes)
    N = g["luces"].shape[0]
    dt = cfg.dt_h

    pasos_dia = int(round(24.0 / dt))
    n_pasos = cfg.dias * pasos_dia
    inicio_eval = n_pasos - cfg.dias_eval * pasos_dia
    pasos_bloque = int(round(HORAS_POR_BLOQUE / dt))
    bloque_de_paso = (np.arange(n_pasos) // pasos_bloque) % N_BLOQUES

    D_h = g["dilucion_d"] / 24.0
    L_cm = g["espesor_cm"]
    volumen_L = cfg.area_m2 * (L_cm / 100.0) * 1000.0
    k = valor(escenario, "luz", "k_extincion")
    i_inhibicion = valor(escenario, "luz", "i_inhibicion")

    X = np.full(N, cfg.biomasa_inicial)
    cosecha_gL = np.zeros(N)
    energia_kwh = np.zeros(N)
    estres = np.zeros(N)
    violacion = np.zeros(N)
    if series:
        s_X = np.zeros((N, n_pasos))
        s_q = np.zeros((N, n_pasos))
        s_Iprom = np.zeros((N, n_pasos))
        s_P = np.zeros((N, n_pasos))

    for i in range(n_pasos):
        q = g["luces"][:, bloque_de_paso[i]]
        r = tasa_crecimiento(q, X, L_cm, cfg.T_C, escenario)
        I_prom = luz_promedio(q, X, L_cm, k)
        P_led = potencia_led(q, cfg.area_m2, escenario)

        # Restricción de biomasa en todo momento (violación relativa, 0 si cumple)
        fuera = (np.maximum(0.0, cfg.X_min - X) / cfg.X_min
                 + np.maximum(0.0, X - cfg.X_max) / cfg.X_max)
        violacion = np.maximum(violacion, fuera)

        if i >= inicio_eval:
            cosecha_gL += D_h * X * dt
            energia_kwh += P_led * dt / 1000.0
            I_estres = I_prom if cfg.base_estres == "promedio" else q
            estres += np.maximum(0.0, I_estres - i_inhibicion) * dt * 3600.0

        if series:
            s_X[:, i], s_q[:, i], s_Iprom[:, i], s_P[:, i] = X, q, I_prom, P_led

        X = np.maximum(X + (r - D_h * X) * dt, 0.0)

    cosechado_g = cosecha_gL * volumen_L
    resultado = ResultadoSimulacion(
        productividad_biomasa=cosecha_gL / cfg.dias_eval,
        energia_por_gramo=energia_kwh / (cosechado_g + cfg.delta_g),
        eficiencia_co2=np.full(N, np.nan),
        estres_luminico=estres,
        violacion=violacion,
        energia_led_kwh=energia_kwh,
        energia_mezclado_kwh=np.zeros(N),
        biomasa_cosechada_g=cosechado_g,
        biomasa_final_gL=X,
    )
    if series:
        resultado.series = {
            "tiempo_h": np.arange(n_pasos) * dt,
            "biomasa": s_X, "luz_incidente": s_q,
            "luz_promedio": s_Iprom, "potencia_led_w": s_P,
        }
    return resultado