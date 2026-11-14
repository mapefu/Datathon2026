"""
agent.py - BatteryAgent para el reto UPC Sostenible (Datathon 2026)

Estrategia: optimización lineal con horizonte móvil de 24 h.
Cada hora el agente resuelve un pequeño problema de programación lineal con las
previsiones de precio y CO2 de las próximas 24 h (que son datos exactos, no estimaciones)
y aplica solo la primera decisión. No usa machine learning: la demanda del campus
(~1.100 kW de media) es mucho mayor que la potencia de la batería (500 kW), así que
casi nunca hay que predecirla.
"""

import numpy as np
from scipy.optimize import linprog


class BatteryAgent:
    def __init__(self):
        # Física de la batería (battery_spec.json)
        self.capacitat_kwh = 2000.0
        self.potencia_nominal_kw = 500.0
        self.eta = 0.90 ** 0.5                 # rendimiento de una carga o de una descarga

        # Pesos de la puntuación: 70% CO2, 30% coste
        self.pes_co2 = 0.70
        self.pes_preu = 0.30

        # Media horaria de (importación de la red * CO2) y de (importación * precio) en los datos
        # de entrenamiento. Sirven para que 1 kg de CO2 y 1 € se midan con la misma vara,
        # porque la puntuación es un porcentaje respecto a lo que gasta el campus sin batería.
        self.ref_co2 = 92.078
        self.ref_cost = 156.789

        # Valor de dejar energía guardada al final de las 24 h (relativo al coste medio por kWh).
        # Cualquier valor entre 0,2 y 0,7 daba la misma puntuación en las pruebas.
        self.kappa = 0.5

        self._tri = {}                         # matrices triangulares ya construidas, por horizonte

    # ------------------------------------------------------------------
    def _finestra(self, obs):
        """Precio y CO2 de la hora actual y las siguientes, sin duplicar la hora actual.
        En el simulador local la lista de 24 valores ya empieza en la hora actual."""
        p = list(obs["preus_properes_24h"])
        c = list(obs["co2_properes_24h"])
        ja_inclou_actual = (
            len(p) > 0 and len(c) > 0
            and abs(p[0] - obs["preu_actual"]) < 1e-9
            and abs(c[0] - obs["co2_actual"]) < 1e-9
        )
        if not ja_inclou_actual:
            p = [obs["preu_actual"]] + p
            c = [obs["co2_actual"]] + c
        return np.array(p, dtype=float), np.array(c, dtype=float)

    def _cost_per_hora(self, preus, co2):
        """Lo malo que es importar 1 kWh de la red en cada hora (mezcla 70/30 normalizada)."""
        return self.pes_co2 * co2 / self.ref_co2 + self.pes_preu * preus / self.ref_cost

    def _optimitzar(self, a, soc_kwh):
        """Resuelve el plan de carga/descarga y devuelve la consigna de esta hora (kW)."""
        H = len(a)
        if H not in self._tri:
            self._tri[H] = np.tril(np.ones((H, H)))
        T = self._tri[H]
        eta = self.eta

        lam = self.kappa * a.mean()            # valor de un kWh guardado al final del horizonte
        c_carrega = a - lam * eta              # coste de las variables de carga
        c_descarrega = -a + lam / eta          # coste de las variables de descarga
        obj = np.concatenate([c_carrega, c_descarrega])

        # SOC al final de cada hora k: s0 + T @ (eta*carga - descarga/eta), entre 0 y capacidad
        M = np.hstack([eta * T, -T / eta])
        A_ub = np.vstack([M, -M])
        b_ub = np.concatenate([np.full(H, self.capacitat_kwh - soc_kwh), np.full(H, soc_kwh)])
        bounds = [(0.0, self.potencia_nominal_kw)] * (2 * H)

        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.status != 0:
            raise RuntimeError("LP sin solución")
        return float(res.x[0] - res.x[H])

    def _regles(self, a):
        """Plan B si el optimizador falla: cargar en las horas más baratas y descargar en las más caras."""
        if a[0] <= np.percentile(a, 25):
            return self.potencia_nominal_kw
        if a[0] >= np.percentile(a, 75):
            return -self.potencia_nominal_kw
        return 0.0

    # ------------------------------------------------------------------
    def act(self, obs: dict) -> float:
        soc_kwh = float(obs["soc_kwh"])

        consigna = 0.0
        try:
            preus, co2 = self._finestra(obs)
            a = self._cost_per_hora(preus, co2)
            if np.all(np.isfinite(a)) and len(a) > 0:
                try:
                    consigna = self._optimitzar(a, soc_kwh)
                except Exception:
                    consigna = self._regles(a)
        except Exception:
            consigna = 0.0

        # Límites físicos: bloque recomendado por la organización (0 infracciones por clipping)
        eta = self.eta
        eps = 1e-4
        espai_disponible_kwh = max(0.0, self.capacitat_kwh - soc_kwh)
        energia_disponible_kwh = max(0.0, soc_kwh)

        max_carrega = max(0.0, min(self.potencia_nominal_kw, espai_disponible_kwh / eta - eps))
        max_descarrega = max(0.0, min(self.potencia_nominal_kw, energia_disponible_kwh * eta - eps))

        return float(np.clip(consigna, -max_descarrega, max_carrega))
