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

Robustesa (no canvia res si les dades són exactes i completes):
  - λ online: el quocient CO2_base/Cost_base s'actualitza amb les hores ja viscudes
    (prior de train ≈ 0,59 kg/€ amb pes d'una setmana).
  - Previsions amb error: s'estima l'error real de cada horitzó comparant el que es
    va preveure amb el que després passa. Si l'error és zero, es fa servir la
    previsió tal qual. Si no, per a cada horitzó es tria entre la previsió més
    recent i la combinació de totes les previsions d'aquella hora, segons quina
    ha encertat més fins ara (vegeu _Previsor).
  - NaN, claus que falten, tipus inesperats (arrays, Series) o qualsevol excepció →
    regla de reserva o 0 kW, i el pas següent torna a funcionar amb normalitat. Si falta
    el SOC, es fa servir soc_pct/soc_percent o un model intern de la física; si no n'hi ha
    cap, 0 kW. Sempre es
    retalla als límits físics: 0 infraccions.
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

# λ = (0.30/0.70)·(CO2_base/Cost_base). Prior de train: CO2_base/Cost_base ≈ 0,59 kg/€.
RATIO_PRIOR = 0.59
PRIOR_COST_EUR = 168 * 1100.0 * 0.138  # pes del prior ≈ una setmana de consum típic
LAMBDA = 0.30 / 0.70 * RATIO_PRIOR
V_QUANTILE = 0.40

# Estimació de l'error de les previsions
EWMA_ALPHA = 0.02     # memòria ≈ 50 observacions per horitzó
MIN_OBS_ERR = 24      # observacions mínimes abans de fiar-se de la variància estimada
VAR_EXACTA = 1e-12    # per sota d'això, la previsió es considera exacta

# Pressupost de temps: el servidor talla a 60 s. Es mesura el temps real des de la primera
# crida (inclou l'entorn, com al servidor). Si ens hi acostem, passem a una regla barata
# (sense LP) per no perdre el lliurament.
TIME_BUDGET_S = 40.0


def _serie(x) -> np.ndarray:
    """Converteix llista/array/Series/None en un array de floats (NaN si no és numèric). No llança mai."""
    if x is None or isinstance(x, (str, bytes, dict)):
        return np.empty(0)
    try:
        return np.array([_f(v) for v in x], dtype=float)
    except TypeError:
        return np.empty(0)


def _f(x, default=float("nan")):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


class _Previsor:
    """Millora les previsions d'una sèrie (preu o CO2) segons l'error que realment tenen.

    Per a cada hora futura hi ha diverses previsions (la d'ara i les d'hores anteriors).
    Es comparen dos estimadors i, per a cada horitzó k, es fa servir el que ha tingut
    menys error real fins ara:
      - "fresca": la previsió més recent (la que dona obs ara mateix);
      - "combinada": mitjana de totes les previsions ponderada per 1/variància de l'horitzó.
    La combinació guanya si els errors de previsions successives són independents i
    perd si són persistents; mesurant-ho no cal suposar res. Si les previsions són
    exactes (error 0) es fa servir la fresca tal qual.
    """

    def __init__(self, n_h: int = 24):
        self.hist = []                       # valor real a cada pas
        self.pendents = {}                   # pas objectiu -> [(horitzó k, previsió)]
        self.comb_pend = {}                  # pas objectiu -> [(horitzó k, estimació combinada)]
        self.var = np.zeros(n_h + 1)         # error relatiu² (EWMA) de la previsió fresca, per k
        self.var_comb = np.zeros(n_h + 1)    # ídem per a l'estimació combinada
        self.n_obs = np.zeros(n_h + 1, dtype=int)
        self.n_comb = np.zeros(n_h + 1, dtype=int)

    @staticmethod
    def _ewma(v, k, e):
        v[k] = (1 - EWMA_ALPHA) * v[k] + EWMA_ALPHA * e * e

    def observa(self, n: int, actual: float, previsio: np.ndarray) -> float:
        """Registra el valor real del pas n i la nova previsió. Retorna el valor real net."""
        if not math.isfinite(actual):
            # Sense valor actual: el millor substitut és la previsió per a l'hora 0.
            actual = _f(previsio[0]) if len(previsio) else float("nan")
            if not math.isfinite(actual):
                actual = self.hist[-1] if self.hist else 0.0
        while len(self.hist) < n:  # passos perduts per una excepció: s'omplen perquè n quadri
            self.hist.append(self.hist[-1] if self.hist else actual)
        escala = max(abs(actual), 1e-3)
        for k, v in self.pendents.pop(n, []):
            self._ewma(self.var, k, (v - actual) / escala)
            self.n_obs[k] += 1
        for k, v in self.comb_pend.pop(n, []):
            self._ewma(self.var_comb, k, (v - actual) / escala)
            self.n_comb[k] += 1
        self.hist.append(actual)
        for k in range(1, len(previsio)):
            v = previsio[k]
            if math.isfinite(v):
                self.pendents.setdefault(n + k, []).append((k, float(v)))
        return actual

    def finestra(self, n: int, H: int) -> np.ndarray:
        out = np.empty(H)
        out[0] = self.hist[n]
        for k in range(1, H):
            cands = self.pendents.get(n + k, [])
            fresca = next((v for kk, v in cands if kk == k), None)
            fiable = self.n_obs[k] >= MIN_OBS_ERR
            if fresca is not None and (not fiable or self.var[k] < VAR_EXACTA):
                out[k] = fresca
                continue
            # Estimació combinada (ponderada per 1/variància de cada horitzó).
            num = den = 0.0
            for kk, v in cands:
                if self.n_obs[kk] >= MIN_OBS_ERR and self.var[kk] > 0:
                    w = 1.0 / self.var[kk]
                    num += w * v
                    den += w
            if den > 0:
                comb = num / den
            else:  # sense variàncies fiables: la previsió més recent que hi hagi per a aquella hora
                comb = fresca if fresca is not None else (min(cands)[1] if cands else None)
            if comb is not None:
                self.comb_pend.setdefault(n + k, []).append((k, comb))
            usa_comb = (
                comb is not None
                and self.n_comb[k] >= MIN_OBS_ERR
                and (fresca is None or self.var_comb[k] < self.var[k])
            )
            if usa_comb or fresca is None:
                out[k] = comb if comb is not None else out[k - 1]
            else:
                out[k] = fresca
        return out


class BatteryAgent:
    def __init__(self, lam=None, v_quantile: float = V_QUANTILE, robust: bool = True):
        self.lam_fix = lam  # si es dona, desactiva λ online (per a experiments)
        self.v_quantile = v_quantile
        self.robust = robust
        self._lp_cache = {}
        self._t_inici = None
        self.n = 0
        self.preu = _Previsor()
        self.co2 = _Previsor()
        self.s_co2 = 0.0
        self.s_cost = 0.0
        self._soc_model = None

    # Matrius de l'LP per a un horitzó H (només canvien el cost i el SOC inicial).
    # Variables: [c_0..c_{H-1}, d_0..d_{H-1}, s_0..s_{H-1}], s_k = SOC al final de l'hora k.
    def _lp(self, H: int):
        if H not in self._lp_cache:
            I = sparse.identity(H, format="csr")
            D = I - sparse.eye(H, k=-1, format="csr")
            self._lp_cache[H] = sparse.hstack([-ETA * I, I / ETA, D], format="csr")
        return self._lp_cache[H]

    def _lambda(self) -> float:
        if self.lam_fix is not None:
            return self.lam_fix
        if not self.robust:
            return LAMBDA
        ratio = (RATIO_PRIOR * PRIOR_COST_EUR + self.s_co2) / (PRIOR_COST_EUR + self.s_cost)
        return 0.30 / 0.70 * ratio

    def act(self, obs: dict) -> float:
        if self._t_inici is None:
            self._t_inici = time.perf_counter()
        soc = self._soc(obs)
        if soc is None:  # sense SOC no es poden garantir 0 infraccions: no actuar
            self._soc_model = None
            return 0.0
        try:
            p = self._decideix(obs, soc)
        except Exception:
            p = 0.0
        # Seguretat física: mai demanar més del que permeten inversor i SOC.
        p = _f(p, 0.0)
        max_c = max(0.0, min(P_MAX, (CAP - soc) / ETA - MARGE))
        max_d = max(0.0, min(P_MAX, soc * ETA - MARGE))
        p = max(-max_d, min(max_c, p))
        p = 0.0 if abs(p) < 1e-3 else float(p)
        self._soc_model = soc + (ETA * p if p > 0 else p / ETA)
        return p

    def _soc(self, obs):
        """SOC en kWh: soc_kwh → soc_pct → soc_percent → model intern → None."""
        if not isinstance(obs, dict):
            obs = {}
        for key, esc in (("soc_kwh", 1.0), ("soc_pct", CAP), ("soc_percent", CAP / 100)):
            v = _f(obs.get(key)) * esc
            if math.isfinite(v):
                return min(max(v, 0.0), CAP)
        return self._soc_model

    def _decideix(self, obs: dict, soc: float) -> float:
        n = self.n
        self.n += 1
        prev_p = _serie(obs.get("preus_properes_24h"))
        prev_c = _serie(obs.get("co2_properes_24h"))
        H = min(len(prev_p), len(prev_c))

        # Càrrega neta de l'hora anterior → λ online i tope de descàrrega.
        dades = obs.get("noves_dades_5min")
        if not isinstance(dades, dict):
            dades = {}
        net_prev = _f(dades.get("ultim_consum_kw"), 1e9) - _f(dades.get("ultima_solar_kw"), 0.0)
        if n >= 1 and net_prev < 1e8 and len(self.co2.hist) >= n and len(self.preu.hist) >= n:
            net = max(net_prev, 0.0)
            self.s_co2 += net * self.co2.hist[n - 1]
            self.s_cost += net * self.preu.hist[n - 1]

        self.preu.observa(n, _f(obs.get("preu_actual")), prev_p[:H])
        self.co2.observa(n, _f(obs.get("co2_actual")), prev_c[:H])
        if H == 0:
            return 0.0

        if self.robust:
            preus = self.preu.finestra(n, H)
            co2 = self.co2.finestra(n, H)
        else:
            preus, co2 = prev_p[:H], prev_c[:H]
        w = co2 + self._lambda() * preus

        # Finestra incompleta → l'avaluació acaba dins l'horitzó: l'energia final no val res.
        v = float(np.quantile(w, self.v_quantile)) if H >= 24 else 0.0

        # Si el campus consumeix menys de 500 kW (diumenges a migdia o caigudes de dades),
        # l'excés de descàrrega s'aboca a la xarxa sense valor.
        d0_max = min(P_MAX, max(0.0, net_prev))

        p = None
        if time.perf_counter() - self._t_inici < TIME_BUDGET_S:
            cost = np.concatenate([w, -w, np.zeros(H)])
            cost[-1] -= v
            b_eq = np.zeros(H)
            b_eq[0] = soc
            bounds = [(0.0, P_MAX)] * (2 * H) + [(0.0, CAP)] * H
            bounds[H] = (0.0, d0_max)
            sol = linprog(cost, A_eq=self._lp(H), b_eq=b_eq, bounds=bounds, method="highs")
            if sol.success:
                p = float(sol.x[0] - sol.x[H])
        if p is None:  # Regla de reserva: carregar a les hores més netes/barates, descarregar a les pitjors.
            rank = float(np.mean(w <= w[0]))
            p = P_MAX if rank <= 0.25 else (-d0_max if rank >= 0.75 else 0.0)
        return p
