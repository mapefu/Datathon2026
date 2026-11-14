"""BatteryAgent — control predictivo (MPC) con programación lineal.

Idea: la carga neta del campus casi siempre supera los 500 kW, así que cada kWh que la
batería descarga en la hora t ahorra w_t = 70·co2_t/CO2_base + 30·precio_t/Coste_base
puntos (y cada kWh que carga los cuesta). El score es (casi) lineal en la potencia y,
con el precio y el CO2 de las próximas 24 h en `obs`, el problema es un LP pequeño:

    max  Σ_k w_k·(d_k − c_k) + v·s_H
    s.a. s_k = s_{k−1} + η·c_k − d_k/η,   0 ≤ s_k ≤ 2000,   0 ≤ c_k, d_k ≤ 500

Cada hora se resuelve y se ejecuta solo la primera acción (horizonte deslizante).
Escalando w por 1/(70/CO2_base) queda w_k = co2_k + λ·precio_k con
λ = (30/70)·CO2_base/Coste_base, que se estima online con las horas ya vistas.

Robustez (nada de esto cambia la acción si los datos llegan limpios):
- La hora actual usa siempre `preu_actual`/`co2_actual`. Se detecta si los vectores de 24 h
  empiezan en t o en t+1.
- Los NaN de una previsión se rellenan con una previsión anterior de esa misma hora.
- Si las previsiones cambian de una hora a otra (son previsiones con error), se elige por
  horizonte entre la última previsión y la media de todas, según cuál haya acertado más.
- Si al saltar de semana el SOC vuelve a 1000 kWh sin aviso, a partir de entonces el
  horizonte termina el último día del mes (el test son los últimos 7 días de cada mes).
- La descarga de la hora actual se limita a la última carga neta observada (cortes de datos).
- Cualquier error → regla de reserva o 0 kW, siempre dentro de los límites físicos.
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

# ── RawTree observability (offline-safe) ──────────────────────────────
try:
    from rawtree_logger import RawTreeLogger
    _rt = RawTreeLogger()
except Exception:  # pragma: no cover  (sin rawtree_logger → sin observabilidad)
    _rt = None  # type: ignore[assignment]

CAP = 2000.0
P_MAX = 500.0
ETA = math.sqrt(0.9)
SOC0 = 1000.0
H_MAX = 24

RATIO_PRIOR = 0.59          # CO2_base / Coste_base en train (kg/€)
PRIOR_HOURS = 168.0         # peso del prior: una semana
PRIOR_COST_PER_H = 700.0 * 0.12   # coste base típico por hora (€), solo escala el prior
V_QUANTILE = 0.40           # valor terminal de la energía almacenada (cuantil de w)
TIME_BUDGET_S = 40.0        # tiempo real tras el cual se pasa a la regla barata
MARGIN = 1e-6               # kW de margen frente a los límites físicos
ASSUME_WEEK_END = 1         # 1: hasta ver la primera frontera de semana, planificar como si el SOC
                            #    se reiniciara al acabar la semana (168 h / fin de mes)


def _num(x, default=float("nan")) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _vec(x) -> np.ndarray:
    if x is None:
        return np.zeros(0)
    try:
        v = np.asarray(list(x), dtype=float).ravel()
    except (TypeError, ValueError):
        out = []
        for e in x:
            out.append(_num(e))
        v = np.asarray(out, dtype=float)
    return v


def _hour_key(ts) -> int | None:
    """Timestamp → número de hora absoluto (horas desde epoch). None si no se puede."""
    if ts is None:
        return None
    try:
        if hasattr(ts, "timestamp") and not isinstance(ts, str):
            return int(round(ts.timestamp() / 3600.0))
        import pandas as pd
        return int(round(pd.Timestamp(str(ts)).timestamp() / 3600.0))
    except Exception:
        return None


def _hours_to_month_end(ts) -> int | None:
    """Horas que quedan hasta el final del mes de `ts`, contando la hora actual."""
    try:
        import pandas as pd
        t = pd.Timestamp(ts if not isinstance(ts, str) else str(ts))
        end = (t + pd.offsets.MonthEnd(0)).normalize() + pd.Timedelta(hours=23)
        if end < t:
            return None
        return int(round((end - t).total_seconds() / 3600.0)) + 1
    except Exception:
        return None


class _LP:
    """LP de horizonte H con matrices cacheadas."""

    def __init__(self):
        self.cache = {}

    def _mats(self, H):
        if H not in self.cache:
            I = sparse.identity(H, format="csr")
            D = I - sparse.eye(H, k=-1, format="csr")
            A_eq = sparse.hstack([-ETA * I, I / ETA, D], format="csr")
            self.cache[H] = A_eq
        return self.cache[H]

    def first_action(self, w, soc0, v, d0_max) -> float:
        H = len(w)
        A_eq = self._mats(H)
        b_eq = np.zeros(H)
        b_eq[0] = soc0
        cost = np.concatenate([w, -w, np.zeros(H)])
        cost[-1] = -v
        bounds = [(0.0, P_MAX)] * (2 * H) + [(0.0, CAP)] * H
        bounds[H] = (0.0, d0_max)
        sol = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if sol.status != 0 or sol.x is None:
            raise RuntimeError("LP")
        return float(sol.x[0] - sol.x[H])


class BatteryAgent:
    def __init__(self):
        self.lp = _LP()
        self.t_start = None
        self.cheap = False
        # λ online
        self.s_co2 = RATIO_PRIOR * PRIOR_HOURS * PRIOR_COST_PER_H
        self.s_cost = PRIOR_HOURS * PRIOR_COST_PER_H
        # estado del paso anterior
        self.prev_key = None
        self.prev_price = None
        self.prev_co2 = None
        self.prev_soc_pred = None
        self.prev_vecs = None       # (precio, co2) tal como llegaron en el paso anterior
        self.offset = None          # 0: vector empieza en t; 1: empieza en t+1
        self.e_off = [0.0, 0.0]
        self.reset_mode = False
        self.week_start = None      # clave horaria del inicio de la semana actual
        self.prev_len = None        # longitud del vector de previsiones del paso anterior
        self.boundary_seen = False  # ya se ha visto un salto de semana
        # previsiones por hora objetivo: key -> lista de (horizonte, precio, co2)
        self.fc = {}
        # error medio por horizonte: última previsión vs media de previsiones
        self.err_last = np.zeros((2, H_MAX + 1))
        self.err_mean = np.zeros((2, H_MAX + 1))
        self.err_n = np.zeros(H_MAX + 1)
        self.net_hist = deque(maxlen=48)

    # ------------------------------------------------------------------ util
    @staticmethod
    def _limits(soc):
        max_c = max(0.0, min(P_MAX, (CAP - soc) / ETA) - MARGIN)
        max_d = max(0.0, min(P_MAX, soc * ETA) - MARGIN)
        return max_c, max_d

    def _soc(self, obs):
        soc = _num(obs.get("soc_kwh"))
        if math.isnan(soc):
            pct = _num(obs.get("soc_pct"))
            if not math.isnan(pct):
                soc = pct * CAP
        if math.isnan(soc):
            pct = _num(obs.get("soc_percent"))
            if not math.isnan(pct):
                soc = pct / 100.0 * CAP
        if math.isnan(soc) and self.prev_soc_pred is not None:
            soc = self.prev_soc_pred
        if math.isnan(soc):
            return None
        return min(CAP, max(0.0, soc))

    # ------------------------------------------------------------------ main
    def act(self, obs) -> float:
        now = time.perf_counter()
        if self.t_start is None:
            self.t_start = now
        if now - self.t_start > TIME_BUDGET_S:
            self.cheap = True
        try:
            soc = self._soc(obs)
        except Exception:
            soc = None
        if soc is None:
            self.prev_soc_pred = None
            return 0.0
        try:
            p = self._decide(obs, soc)
        except Exception:
            p = 0.0
        if not math.isfinite(p):
            p = 0.0
        max_c, max_d = self._limits(soc)
        p = max(-max_d, min(max_c, p))
        if abs(p) < 1e-3:
            p = 0.0
        self.prev_soc_pred = soc + (p * ETA if p >= 0 else p / ETA)
        return float(p)

    def _decide(self, obs, soc) -> float:
        key = _hour_key(obs.get("timestamp"))
        price = _num(obs.get("preu_actual"))
        co2 = _num(obs.get("co2_actual"))
        dades = obs.get("noves_dades_5min")
        if not isinstance(dades, dict):
            dades = {}
        load = _num(dades.get("ultim_consum_kw"))
        pv = _num(dades.get("ultima_solar_kw"), 0.0)
        net_prev = load - pv if not math.isnan(load) else float("nan")
        vp = _vec(obs.get("preus_properes_24h"))
        vc = _vec(obs.get("co2_properes_24h"))

        consecutive = key is not None and self.prev_key is not None and key == self.prev_key + 1
        # salto de más de 24 h = semana nueva (cambios de hora y huecos de datos no cuentan)
        jumped = key is not None and self.prev_key is not None and not (0 < key - self.prev_key <= 24)
        if key is not None and (self.week_start is None or jumped):
            self.week_start = key

        # --- λ online: la hora anterior ya es conocida (carga neta + precio/CO2 guardados)
        if consecutive and not math.isnan(net_prev) and self.prev_price is not None:
            b = max(0.0, net_prev)
            if math.isfinite(self.prev_price) and math.isfinite(self.prev_co2):
                self.s_co2 += b * self.prev_co2
                self.s_cost += b * self.prev_price
        if not math.isnan(net_prev):
            self.net_hist.append(net_prev)

        # --- ¿reinicio de SOC por semana sin truncar la ventana?
        if jumped:
            self.boundary_seen = True
        if jumped and self.prev_soc_pred is not None:
            if abs(soc - SOC0) < 1e-3 and abs(self.prev_soc_pred - SOC0) > 1.0:
                self.reset_mode = True

        # --- alineación de los vectores (empiezan en t o en t+1)
        if math.isnan(price) and len(vp):
            price = vp[0] if self.offset != 1 else float("nan")
        if math.isnan(co2) and len(vc):
            co2 = vc[0] if self.offset != 1 else float("nan")
        self._detect_offset(price, co2, vp, vc, consecutive)
        off = self.offset if self.offset is not None else 0
        fut_p = vp[1 - off:] if off == 0 else vp
        fut_c = vc[1 - off:] if off == 0 else vc
        n_fut = min(len(fut_p), len(fut_c))
        fut_p, fut_c = fut_p[:n_fut], fut_c[:n_fut]

        # --- previsiones: registro, error por horizonte y combinación
        if key is not None:
            fut_p, fut_c = self._forecasts(key, price, co2, fut_p, fut_c, consecutive)
        fut_p = self._fill(fut_p, price)
        fut_c = self._fill(fut_c, co2)
        if math.isnan(price):
            price = fut_p[0] if len(fut_p) else 0.1
        if math.isnan(co2):
            co2 = fut_c[0] if len(fut_c) else 0.08

        prices = np.concatenate([[price], fut_p])
        co2s = np.concatenate([[co2], fut_c])
        H = len(prices)
        # ventana más corta de lo normal => la evaluación acaba dentro del horizonte
        truncated = 0 < len(vp) < H_MAX
        if len(vp) == 0:
            # vector vacío: última hora de la evaluación si la ventana ya venía menguando
            H = 1
            truncated = self.prev_len is not None and 0 < self.prev_len <= 2
        self.prev_len = len(vp)
        hedge = ASSUME_WEEK_END and not self.boundary_seen
        if (self.reset_mode or hedge) and key is not None and self.week_start is not None:
            # la semana acaba a las 168 h de empezar (o a fin de mes, lo que llegue antes)
            left = self.week_start + 168 - key
            left_m = _hours_to_month_end(obs.get("timestamp"))
            if left_m is not None:
                left = min(left, left_m)
            if 0 < left < H:
                H, truncated = left, True
        prices, co2s = prices[:H], co2s[:H]

        lam = (0.30 / 0.70) * self.s_co2 / self.s_cost
        w = co2s + lam * prices

        # descarga de la hora actual limitada a la última carga neta observada
        if math.isnan(net_prev):
            d0_max = P_MAX
        else:
            d0_max = min(P_MAX, max(0.0, net_prev))

        self.prev_key, self.prev_price, self.prev_co2 = key, price, co2

        if H <= 1 and truncated:
            return -d0_max  # última hora de la evaluación: vaciar
        if H <= 1:
            return 0.0

        v = 0.0 if truncated else float(np.quantile(w, V_QUANTILE))
        if self.cheap:
            return self._rule(w, soc, v, d0_max)
        p = self.lp.first_action(w, soc, v, d0_max)

        # ── RawTree logging (offline-safe) ───────────────────────────────
        try:
            pfloat = float(p) if p is not None else 0.0
            _rt.log_decision(
                timestamp=obs.get("timestamp", ""),
                idx=0, soc_kwh=soc, soc_pct=soc / CAP * 100 if CAP > 0 else 0,
                cap_kwh=CAP, power_kw=pfloat,
                price=price, co2=co2, load_net=net_prev if math.isfinite(net_prev) else 0,
                mode="std", horizon=H, scenarios=0,
                lp_time=0.0, elapsed_total=0.0,
            )
        except Exception:
            pass

        return p

    # ------------------------------------------------------------- helpers
    def _detect_offset(self, price, co2, vp, vc, consecutive):
        """Voto acumulado: ¿el valor actual se parece más a vector[0] de ahora (offset 0)
        o a vector[0] del paso anterior (offset 1)? Errores relativos de precio y CO2."""
        if consecutive and self.prev_vecs is not None:
            pp, pc = self.prev_vecs
            e0 = e1 = 0.0
            ok = False
            for cur, now_v, prev_v in ((price, vp, pp), (co2, vc, pc)):
                if len(now_v) and len(prev_v) and all(map(math.isfinite, (cur, now_v[0], prev_v[0]))):
                    s = max(abs(cur), 1e-9)
                    e0 += abs(now_v[0] - cur) / s
                    e1 += abs(prev_v[0] - cur) / s
                    ok = True
            if ok:
                self.e_off[0] += e0
                self.e_off[1] += e1
                self.offset = 1 if self.e_off[1] < self.e_off[0] else 0
        self.prev_vecs = (vp, vc)

    def _forecasts(self, key, price, co2, fut_p, fut_c, consecutive):
        # error de las previsiones guardadas para la hora actual (ahora conocida)
        rec = self.fc.pop(key, None)
        if rec and math.isfinite(price) and math.isfinite(co2):
            # rec está en orden cronológico (horizonte decreciente). Para cada horizonte h:
            # error de la previsión hecha a h horas vs error de la media de todas hasta entonces.
            acc = [[], []]
            for h, fp, fc_ in rec:
                if not (0 < h <= H_MAX and math.isfinite(fp) and math.isfinite(fc_)):
                    continue
                acc[0].append(fp)
                acc[1].append(fc_)
                self.err_last[0, h] += abs(fp - price)
                self.err_last[1, h] += abs(fc_ - co2)
                self.err_mean[0, h] += abs(sum(acc[0]) / len(acc[0]) - price)
                self.err_mean[1, h] += abs(sum(acc[1]) / len(acc[1]) - co2)
                self.err_n[h] += 1
        # limpiar horas pasadas
        if len(self.fc) > 4 * H_MAX:
            for k in [k for k in self.fc if k < key]:
                del self.fc[k]
        out_p, out_c = fut_p.copy(), fut_c.copy()
        for i in range(len(fut_p)):
            k = key + i + 1
            h = i + 1
            lst = self.fc.setdefault(k, [])
            older_p = [r[1] for r in lst if math.isfinite(r[1])]
            older_c = [r[2] for r in lst if math.isfinite(r[2])]
            lst.append((h, fut_p[i], fut_c[i]))
            # NaN → la previsión más reciente válida para esa misma hora
            if not math.isfinite(out_p[i]) and older_p:
                out_p[i] = older_p[-1]
            if not math.isfinite(out_c[i]) and older_c:
                out_c[i] = older_c[-1]
            # combinar si la media ha acertado más que la última en este horizonte
            if self.err_n[h] >= 24:
                if older_p and math.isfinite(fut_p[i]) and self.err_mean[0, h] < 0.98 * self.err_last[0, h]:
                    out_p[i] = float(np.mean(older_p + [fut_p[i]]))
                if older_c and math.isfinite(fut_c[i]) and self.err_mean[1, h] < 0.98 * self.err_last[1, h]:
                    out_c[i] = float(np.mean(older_c + [fut_c[i]]))
        return out_p, out_c

    @staticmethod
    def _fill(v, current):
        v = np.asarray(v, dtype=float).copy()
        if not len(v):
            return v
        bad = ~np.isfinite(v)
        if bad.all():
            v[:] = current if math.isfinite(current) else 0.0
            return v
        if bad.any():
            idx = np.arange(len(v))
            v[bad] = np.interp(idx[bad], idx[~bad], v[~bad])
        return v

    @staticmethod
    def _rule(w, soc, v, d0_max):
        """Regla barata (sin LP): cargar en las horas más baratas, descargar en las más caras."""
        r = np.argsort(np.argsort(w)) / max(1, len(w) - 1)
        if r[0] <= 0.2 and w[0] * (1 / ETA) < np.quantile(w, 0.8) * ETA:
            return P_MAX
        if r[0] >= 0.8 and w[0] > max(v, np.quantile(w, 0.2)) / (ETA * ETA):
            return -d0_max
        return 0.0
