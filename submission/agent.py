"""
agent.py — BatteryAgent per al Repte UPC Sostenible (Datathon 2026).

Estratègia: control predictiu (MPC) amb programació lineal.

La càrrega neta del campus és gairebé sempre > 500 kW, de manera que el score és
pràcticament lineal en la potència de la bateria: descarregar 1 kWh a l'hora t
estalvia w_t = 70·co2_t/CO2_base + 30·preu_t/Cost_base punts, i carregar-lo en costa
w_t. Com que obs dona el preu i el CO2 de les properes 24 h, a cada hora:

  1. Calculem w_k ∝ co2_k + λ·preu_k per a la finestra coneguda.
  2. Resolem un LP (24 h) que maximitza Σ w_k·(d_k − c_k) + v·SOC_final.
  3. Executem només la primera hora (horitzó lliscant).

v = valor residual de l'energia que queda a la bateria al final de la finestra
(quantil de w de la finestra). Quan la finestra té < 24 h, s'ha acabat l'avaluació
dins l'horitzó → v = 0 i la bateria es buida.
"""

import math
import time

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

CAP = 2000.0
P_MAX = 500.0
ETA = math.sqrt(0.9)
MARGE = 1e-6  # kW de marge per no tocar mai el límit físic exacte

# λ = (0.30/0.70)·(CO2_base/Cost_base); CO2_base/Cost_base ≈ 0.59 kg/€ a train.
LAMBDA = 0.30 / 0.70 * 0.59
V_QUANTILE = 0.40
# Pressupost de temps: el servidor talla a 60 s. Si ens hi acostem, passem a una regla
# barata (sense LP) per no perdre el lliurament.
TIME_BUDGET_S = 40.0


class BatteryAgent:
    def __init__(self, lam: float = LAMBDA, v_quantile: float = V_QUANTILE):
        self.lam = lam
        self.v_quantile = v_quantile
        self._lp_cache = {}
        self._t_total = 0.0

    # Matrius de l'LP per a un horitzó H (només canvien el cost i el SOC inicial).
    # Variables: [c_0..c_{H-1}, d_0..d_{H-1}, s_0..s_{H-1}], s_k = SOC al final de l'hora k.
    def _lp(self, H: int):
        if H not in self._lp_cache:
            I = sparse.identity(H, format="csr")
            D = I - sparse.eye(H, k=-1, format="csr")
            A_eq = sparse.hstack([-ETA * I, I / ETA, D], format="csr")
            self._lp_cache[H] = A_eq
        return self._lp_cache[H]

    def act(self, obs: dict) -> float:
        soc = float(obs["soc_kwh"])
        preus = np.asarray(obs["preus_properes_24h"], dtype=float)
        co2 = np.asarray(obs["co2_properes_24h"], dtype=float)
        H = min(len(preus), len(co2))
        if H == 0:
            return 0.0
        w = co2[:H] + self.lam * preus[:H]

        # Finestra incompleta → l'avaluació acaba dins l'horitzó: l'energia final no val res.
        v = float(np.quantile(w, self.v_quantile)) if H >= 24 else 0.0

        # Límit de descàrrega útil a l'hora actual: si el campus consumeix menys de 500 kW
        # (diumenges a migdia o caigudes de dades), l'excés s'aboca a la xarxa sense valor.
        dades = obs.get("noves_dades_5min") or {}
        net_prev = float(dades.get("ultim_consum_kw", 1e9)) - float(dades.get("ultima_solar_kw", 0.0))
        d0_max = min(P_MAX, max(0.0, net_prev))

        cost = np.concatenate([w, -w, np.zeros(H)])
        cost[-1] -= v
        b_eq = np.zeros(H)
        b_eq[0] = soc
        bounds = [(0.0, P_MAX)] * (2 * H) + [(0.0, CAP)] * H
        bounds[H] = (0.0, d0_max)

        p = None
        if self._t_total < TIME_BUDGET_S:
            t0 = time.perf_counter()
            sol = linprog(cost, A_eq=self._lp(H), b_eq=b_eq, bounds=bounds, method="highs")
            self._t_total += time.perf_counter() - t0
            if sol.success:
                p = float(sol.x[0] - sol.x[H])
        if p is None:  # Regla de reserva: carregar a les hores més netes/barates, descarregar a les pitjors.
            rank = float(np.mean(w <= w[0]))
            p = P_MAX if rank <= 0.25 else (-d0_max if rank >= 0.75 else 0.0)

        # Seguretat física: mai demanar més del que permeten inversor i SOC.
        max_c = max(0.0, min(P_MAX, (CAP - soc) / ETA - MARGE))
        max_d = max(0.0, min(P_MAX, soc * ETA - MARGE))
        p = max(-max_d, min(max_c, p))
        if abs(p) < 1e-3:
            p = 0.0
        return float(p)
