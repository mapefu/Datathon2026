"""
agent.py — BatteryAgent per al Repte UPC Sostenible (Datathon 2026).

Estratègia: control predictiu (MPC) amb programació lineal i model de campus.

Per què un LP:
    El consum del campus supera la generació solar pràcticament sempre
    (P_xarxa > 0), així que la puntuació és lineal en l'energia intercanviada
    amb la bateria:
        Score = 100 - Σ_t w_t · P_xarxa,t
        w_t   = 70·co2_t/CO2_base + 30·preu_t/Cost_base
    Minimitzar Σ w_t·P_xarxa,t és, per tant, exactament maximitzar el Score.
    `obs` inclou el preu i el CO2 exactes de les properes 24 h, de manera que
    cada hora resolem un LP amb la dinàmica exacta del simulador (η = √0,90,
    capacitat 2.000 kWh, potència ±500 kW) i executem només la primera hora
    (horitzó lliscant).

Què afegeix respecte d'un MPC ingenu:
    1. Model de càrrega (consum del campus) après en línia per (dia, hora) i
       model físic de generació solar (geometria solar de Barcelona + factor
       de núvol après). Serveixen per estimar la càrrega neta futura i detectar
       hores amb excedent solar, on carregar és GRATIS (P_xarxa = 0) i
       descarregar no val res (l'excedent s'aboca a 0 €).
    2. Restricció de xarxa explícita: P_xarxa,t >= max(0, net_t + c_t - d_t).
       Així el LP no malgasta energia descarregant per sota del consum real ni
       deixa passar excedents gratis.
    3. Valor terminal de l'energia emmagatzemada (quantil de w) per no buidar
       la bateria de forma miop al final de la finestra de 24 h.
    4. λ = (0,30/0,70)·(CO2_base/Cost_base) après en línia amb un prior sòlid,
       de manera que el pes relatiu de CO2 i preu s'adapta al període avaluat.
    5. Previsions robustes: es mesura l'error real de cada horitzó i es combinen
       les previsions si el servidor les lliura amb soroll.
    6. 0 infraccions garantides: retall físic amb marge abans de retornar.

Restriccions de seguretat: només numpy, scipy, math i time (permeses).
"""

import math
import time
from datetime import datetime, timedelta

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# Física de la bateria (battery_spec.json)
# ---------------------------------------------------------------------------
CAP = 2000.0                 # kWh
P_MAX = 500.0                # kW
ETA = math.sqrt(0.90)        # η_c = η_d = √0,90
MARGE = 1.0e-3               # kW de marge per no tocar mai el límit exacte

CAP_NET = 5000.0             # cota superior de la càrrega neta (kW) per seguretat numèrica
LOAD_MIN_VALID = 100.0       # per sota d'això, el sensor de consum es considera caigut

# ---------------------------------------------------------------------------
# Ponderació CO2 / cost
# ---------------------------------------------------------------------------
# Score = 0,70·ΔCO2 + 0,30·ΔCost ⇒ w_t ∝ co2_t + λ·preu_t
# amb λ = (0,30/0,70)·(CO2_base/Cost_base). El quocient de train és ≈ 0,587 kg/€.
RATIO_PRIOR = 0.5873
PRIOR_COST_EUR = 168 * 1100.0 * 0.138   # pes del prior ≈ una setmana de consum
LAMBDA_PRIOR = 0.30 / 0.70 * RATIO_PRIOR

V_QUANTILE = 0.50            # valor terminal: quantil de w de la finestra

# Previsions (error per horitzó)
EWMA_ALPHA = 0.02
MIN_OBS_ERR = 24
VAR_EXACTA = 1e-12

# Models de campus
LOAD_ALPHA_DH = 0.25         # EWMA per (dia de la setmana, hora)
LOAD_ALPHA_H = 0.10          # EWMA per hora del dia
LOAD_ALPHA_LEVEL = 0.02      # EWMA del nivell global
PV_ALPHA = 0.10              # EWMA del factor de núvol
PV_PRIOR = 110.0             # kW de pic aproximat de la planta (train)
LAT_BARCELONA = 41.3869
LON_BARCELONA = 2.1124

TIME_BUDGET_S = 45.0         # el servidor talla a 60 s

# Mode de model de càrrega neta:
#   "all"  -> estima la càrrega neta de tot l'horitzó (detecta excedents)
#   "hour0"-> només la primera hora (persistència); la resta, sense restricció
NET_MODE = "all"
NET_MARGIN = 0.0             # kW de marge conservador a la càrrega neta estimada


def _f(x, default=float("nan")):
    """Conversió tolerant a float (NaN si no és vàlid)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _parse_ts(ts):
    """Retorna un datetime o None si no es pot interpretar."""
    try:
        # pandas.Timestamp i datetime
        return datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)
    except Exception:
        pass
    try:
        s = str(ts).replace("Z", "+00:00")
        return datetime.fromisoformat(s[:19])
    except Exception:
        return None


class _Previsor:
    """Millora les previsions (preu o CO2) segons l'error real de cada horitzó.

    Si les previsions són exactes (error ≈ 0) es fa servir la previsió fresca tal
    qual; si no, per a cada horitzó es tria entre la previsió més recent i la
    combinació ponderada per 1/variància de totes les previsions disponibles.
    """

    def __init__(self, n_h: int = 24):
        self.hist = []
        self.pendents = {}
        self.comb_pend = {}
        self.var = np.zeros(n_h + 1)
        self.var_comb = np.zeros(n_h + 1)
        self.n_obs = np.zeros(n_h + 1, dtype=int)
        self.n_comb = np.zeros(n_h + 1, dtype=int)

    @staticmethod
    def _ewma(v, k, e):
        v[k] = (1.0 - EWMA_ALPHA) * v[k] + EWMA_ALPHA * e * e

    def reset_pending(self):
        """Descarta previsions pendents (canvi de bloc temporal al test)."""
        self.pendents.clear()
        self.comb_pend.clear()

    def observa(self, n: int, actual: float, previsio: np.ndarray) -> float:
        if not math.isfinite(actual):
            actual = _f(previsio[0]) if len(previsio) else float("nan")
            if not math.isfinite(actual):
                actual = self.hist[-1] if self.hist else 0.0
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
            num = den = 0.0
            for kk, v in cands:
                if self.n_obs[kk] >= MIN_OBS_ERR and self.var[kk] > 0:
                    w = 1.0 / self.var[kk]
                    num += w * v
                    den += w
            comb = num / den if den > 0 else fresca
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
    def __init__(self, lam=None, v_quantile: float = V_QUANTILE, robust: bool = True,
                 net_mode: str = NET_MODE):
        self.lam_fix = lam
        self.v_quantile = v_quantile
        self.robust = robust
        self.net_mode = net_mode

        self._lp_cache = {}
        self._t_total = 0.0
        self.n = 0
        self.preu = _Previsor()
        self.co2 = _Previsor()
        self.s_co2 = 0.0
        self.s_cost = 0.0

        # Models de campus
        self.load_dh = {}
        self.load_h = {}
        self.load_level = None
        self.k_pv = PV_PRIOR
        self.pv_n = 0
        self.prev_ts = None

    # ------------------------------------------------------------------ LP
    def _lp(self, H: int):
        """Matrius de restriccions del LP complet (c, d, g, s)."""
        if ("full", H) not in self._lp_cache:
            I = sparse.identity(H, format="csr")
            D = I - sparse.eye(H, k=-1, format="csr")
            Z = sparse.csr_matrix((H, H))
            A_eq = sparse.hstack([-ETA * I, I / ETA, Z, D], format="csr")
            # g - c + d >= net  ⇔  c - d - g <= -net
            A_ub = sparse.hstack([I, -I, -I, Z], format="csr")
            self._lp_cache[("full", H)] = (A_eq, A_ub)
        return self._lp_cache[("full", H)]

    def _lp_simple(self, H: int):
        """Matrius del LP reduït (c, d, s), equivalent quan la xarxa no buida mai."""
        if ("simple", H) not in self._lp_cache:
            I = sparse.identity(H, format="csr")
            D = I - sparse.eye(H, k=-1, format="csr")
            self._lp_cache[("simple", H)] = sparse.hstack(
                [-ETA * I, I / ETA, D], format="csr")
        return self._lp_cache[("simple", H)]

    def _lambda(self) -> float:
        if self.lam_fix is not None:
            return self.lam_fix
        if not self.robust:
            return LAMBDA_PRIOR
        ratio = (RATIO_PRIOR * PRIOR_COST_EUR + self.s_co2) / (PRIOR_COST_EUR + self.s_cost)
        return 0.30 / 0.70 * ratio

    # -------------------------------------------------------------- models
    @staticmethod
    def _clearsky(dt) -> float:
        """Índex de claror (0..1) per a Barcelona a l'hora `dt` (mig de l'hora)."""
        try:
            doy = dt.timetuple().tm_yday
            x = 2.0 * math.pi * (doy + 284.0) / 365.0
            decl = math.radians(23.45 * math.sin(x))
            ha = math.radians(15.0 * (dt.hour + 0.5 - 12.0))
            lat = math.radians(LAT_BARCELONA)
            sin_elev = (math.sin(lat) * math.sin(decl)
                        + math.cos(lat) * math.cos(decl) * math.cos(ha))
            return max(0.0, sin_elev)
        except Exception:
            return 0.0

    def _update_models(self, obs, ts_dt, hour, dow, n):
        """Actualitza els models amb la telemetria de l'última hora."""
        dades = obs.get("noves_dades_5min") or {}
        last_load = _f(dades.get("ultim_consum_kw"), float("nan"))
        last_pv = _f(dades.get("ultima_solar_kw"), float("nan"))

        # Salt temporal? (canvi de bloc del test)
        gap_ok = True
        gap = None
        if ts_dt is not None and self.prev_ts is not None:
            gap = (ts_dt - self.prev_ts).total_seconds() / 3600.0
            gap_ok = abs(gap - 1.0) < 0.01
            if not gap_ok:
                # Les previsions pendents pertanyen al bloc anterior: fora.
                self.preu.reset_pending()
                self.co2.reset_pending()
        if ts_dt is not None:
            self.prev_ts = ts_dt

        # Càrrega neta per a la primera hora: persistència de la dada mesurada.
        # Si el sensor està caigut (load≈0), la persistència val 0 i evita
        # descarregar contra una xarxa que el simulador veu buida.
        # Si hi ha un salt temporal, l'última dada és del bloc anterior i no val.
        net_prev = float("nan")
        if gap_ok and math.isfinite(last_load) and math.isfinite(last_pv):
            net_prev = max(0.0, last_load - last_pv)
        if math.isfinite(last_load) and last_load >= LOAD_MIN_VALID and gap_ok:
            # La mesura correspon a l'hora anterior; aprenem només dades fiables.
            if ts_dt is not None:
                try:
                    prev = ts_dt - timedelta(hours=1)
                    dh, hh = prev.weekday(), prev.hour
                except Exception:
                    dh, hh = (dow - 1) % 7, (hour - 1) % 24
            else:
                dh, hh = (dow - 1) % 7, (hour - 1) % 24
            self._ewma_d(self.load_dh, (dh, hh), last_load, LOAD_ALPHA_DH)
            self._ewma_d(self.load_h, hh, last_load, LOAD_ALPHA_H)
            self.load_level = (last_load if self.load_level is None
                               else (1 - LOAD_ALPHA_LEVEL) * self.load_level
                               + LOAD_ALPHA_LEVEL * last_load)
            # Factor de núvol (PV / claror), només amb claror suficient.
            if ts_dt is not None and math.isfinite(last_pv):
                cs = self._clearsky(ts_dt - timedelta(hours=1))
                if cs > 0.05 and last_pv >= 0.0:
                    obs_k = last_pv / cs
                    self.k_pv = (1 - PV_ALPHA) * self.k_pv + PV_ALPHA * obs_k
                    self.pv_n += 1
        return net_prev

    @staticmethod
    def _ewma_d(d, key, value, alpha):
        if key in d:
            d[key] = (1.0 - alpha) * d[key] + alpha * value
        else:
            d[key] = value

    def _net_forecast(self, ts_dt, hour, dow, H, net_prev):
        """Càrrega neta estimada (kW) per a les properes H hores."""
        out = np.full(H, CAP_NET, dtype=float)
        last_known = net_prev if math.isfinite(net_prev) else None

        for k in range(H):
            if k == 0:
                if last_known is not None:
                    out[0] = min(CAP_NET, max(0.0, last_known) + NET_MARGIN)
                elif self.load_level is not None:
                    out[0] = min(CAP_NET, max(0.0, self.load_level) + NET_MARGIN)
                continue
            if self.net_mode == "hour0":
                out[k] = CAP_NET
                continue
            if ts_dt is not None:
                try:
                    t = ts_dt + timedelta(hours=k)
                    dh, hh = t.weekday(), t.hour
                except Exception:
                    dh, hh, t = (dow + (hour + k) // 24) % 7, (hour + k) % 24, None
            else:
                dh, hh, t = (dow + (hour + k) // 24) % 7, (hour + k) % 24, None

            load_hat = None
            if (dh, hh) in self.load_dh:
                load_hat = self.load_dh[(dh, hh)]
            elif hh in self.load_h:
                load_hat = self.load_h[hh]
            elif self.load_level is not None:
                load_hat = self.load_level
            elif last_known is not None:
                load_hat = last_known + (self.k_pv * self._clearsky(t) if t else 0.0)

            if load_hat is None:
                continue
            pv_hat = 0.0
            if t is not None:
                pv_hat = max(0.0, self.k_pv * self._clearsky(t))
            out[k] = min(CAP_NET, max(0.0, load_hat - pv_hat) + NET_MARGIN)
        return out

    # --------------------------------------------------------------- act
    def act(self, obs: dict) -> float:
        if not isinstance(obs, dict):
            return 0.0
        soc = _f(obs.get("soc_kwh"))
        if not math.isfinite(soc):
            # Fallback si el SOC no arriba en kWh.
            pct = _f(obs.get("soc_pct"))
            if not math.isfinite(pct):
                pct100 = _f(obs.get("soc_percent"))
                pct = pct100 / 100.0 if math.isfinite(pct100) else 0.0
            soc = pct * CAP
        soc = min(max(soc, 0.0), CAP)
        try:
            p = self._decideix(obs, soc)
        except Exception:
            p = 0.0
        p = _f(p, 0.0)
        # Retall físic: mai més del que permet l'inversor ni el SOC.
        max_c = max(0.0, min(P_MAX, (CAP - soc) / ETA - MARGE))
        max_d = max(0.0, min(P_MAX, soc * ETA - MARGE))
        p = max(-max_d, min(max_c, p))
        return 0.0 if abs(p) < 1e-3 else float(p)

    def _decideix(self, obs: dict, soc: float) -> float:
        n = self.n
        self.n += 1
        hour = int(_f(obs.get("hour"), 0)) % 24
        dow = int(_f(obs.get("day_of_week"), 0)) % 7
        ts_dt = _parse_ts(obs.get("timestamp"))

        prev_p = np.array([_f(x) for x in (obs.get("preus_properes_24h") or [])], dtype=float)
        prev_c = np.array([_f(x) for x in (obs.get("co2_properes_24h") or [])], dtype=float)
        H = min(len(prev_p), len(prev_c))
        if H == 0:
            return 0.0

        net_prev = self._update_models(obs, ts_dt, hour, dow, n)

        # λ online amb les hores ja viscudes.
        if n >= 1 and math.isfinite(net_prev):
            self.s_co2 += net_prev * self.co2.hist[n - 1]
            self.s_cost += net_prev * self.preu.hist[n - 1]

        self.preu.observa(n, _f(obs.get("preu_actual")), prev_p[:H])
        self.co2.observa(n, _f(obs.get("co2_actual")), prev_c[:H])

        if self.robust:
            preus = self.preu.finestra(n, H)
            co2_v = self.co2.finestra(n, H)
        else:
            preus, co2_v = prev_p[:H], prev_c[:H]

        w = co2_v + self._lambda() * preus
        net_hat = self._net_forecast(ts_dt, hour, dow, H, net_prev)

        # Valor terminal (l'energia que quedi al final de la finestra no és gratuïta).
        v = float(np.quantile(w, self.v_quantile)) if H >= 24 else 0.0

        p = None
        if self._t_total < TIME_BUDGET_S:
            t0 = time.perf_counter()
            # Camí ràpid: si la càrrega neta estimada no baixa mai de la potència
            # de l'inversor, el LP reduït (sense variables de xarxa) és exacte.
            simple = bool(np.all(net_hat[1:] >= P_MAX + 1.0))
            if simple:
                A_eq = self._lp_simple(H)
                c_obj = np.concatenate([w, -w, np.zeros(H)])
                c_obj[2 * H + H - 1] -= v
                b_eq = np.zeros(H)
                b_eq[0] = soc
                bounds = [(0.0, P_MAX)] * (2 * H) + [(0.0, CAP)] * H
                if net_hat[0] < P_MAX:
                    bounds[H] = (0.0, max(0.0, float(net_hat[0])))
                sol = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                              method="highs")
            else:
                A_eq, A_ub = self._lp(H)
                c_obj = np.concatenate([np.zeros(H), np.zeros(H), w, np.zeros(H)])
                c_obj[3 * H + H - 1] -= v
                b_eq = np.zeros(H)
                b_eq[0] = soc
                bounds = ([(0.0, P_MAX)] * (2 * H)
                          + [(0.0, None)] * H
                          + [(0.0, CAP)] * H)
                sol = linprog(c_obj, A_ub=A_ub, b_ub=-net_hat, A_eq=A_eq,
                              b_eq=b_eq, bounds=bounds, method="highs")
            self._t_total += time.perf_counter() - t0
            if sol.success:
                p = float(sol.x[0] - sol.x[H])

        if p is None:
            # Regla de reserva: carregar a les hores més netes/barates, descarregar
            # a les pitjors, sense superar la càrrega neta coneguda.
            d0_max = min(P_MAX, max(0.0, net_prev) if math.isfinite(net_prev) else P_MAX)
            w0 = float(w[0])
            if not math.isfinite(w0):
                return 0.0
            rank = float(np.mean(w <= w0))
            p = P_MAX if rank <= 0.25 else (-d0_max if rank >= 0.75 else 0.0)
        return p
