"""RawTree logger — envía las decisiones del agente a RawTree (Tinybird).

API HTTP:  https://api.rawtree.com
  POST /v1/tables/{table}   → ingesta de JSON
  POST /v1/query            → consulta SQL

Variables de entorno necesarias:
  RAWTREE_API_KEY  — API key de RawTree (se obtiene en rawtree.com)

Si la variable no está configurada, se usa un modo offline
(en memoria) para que el agente funcione igual de local.
"""

from __future__ import annotations

import datetime
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx

API_BASE = "https://api.rawtree.com"
_TIMEOUT = 30.0


# ── modo offline (fallback cuando no hay API key) ──────────────────────────

class _OfflineStore:
    """Buffer en memoria que simula RawTree mientras no se autentica."""

    def __init__(self) -> None:
        self.tables: Dict[str, List[dict]] = {}

    def ingest(self, table: str, rows: List[dict]) -> dict:
        self.tables.setdefault(table, []).extend(rows)
        return {"ingested": len(rows)}

    def query(self, sql: str) -> List[dict]:
        """Consulta SQL muy básica sobre el buffer local."""
        import re
        sql_u = sql.upper()

        # Extraer FROM table_name
        m = re.search(r"\bFROM\s+(\w+)", sql_u)
        if not m:
            return []
        tbl = m.group(1).lower()
        data = self.tables.get(tbl, [])

        # Extraer WHERE (ignorado aquí para la demo)
        # Extraer SELECT * o columnas
        sel = sql_u.split("SELECT")[1].split("FROM")[0].strip()
        count_m = re.search(r"\bCOUNT\b", sel)

        # Orden
        order_m = re.search(r"ORDER\s+BY\s+(\w+)(\s+(ASC|DESC))?", sql_u)
        sort_col: Optional[str] = None
        sort_rev = False
        if order_m:
            sort_col = order_m.group(1).lower()
            sort_rev = order_m.group(3) == "DESC"

        # Limit
        limit = 50
        lm = re.search(r"LIMIT\s+(\d+)", sql_u)
        if lm:
            limit = int(lm.group(1))

        # Filtrar y ordenar
        results = list(data)
        if sort_col and sort_col in (results[0] if results else {}):
            results.sort(key=lambda r: r.get(sort_col, 0) or 0, reverse=sort_rev)

        if count_m:
            return [{"count": len(results)}]
        return results[:limit]


# ── cliente principal ───────────────────────────────────────────────────────

class RawTreeLogger:
    """Envía decisiones de batería a RawTree y permite consultar el histórico."""

    def __init__(self, api_key: str | None = None, table_prefix: str = "datathon_battery") -> None:
        self.api_key = api_key or os.environ.get("RAWTREE_API_KEY")
        self.table_prefix = table_prefix
        self.online = bool(self.api_key)
        self._store = _OfflineStore()  # siempre disponible (offline fallback)
        self._client: httpx.Client | None = None
        self.table_decisions = f"{table_prefix}_decisions"
        self.table_scoring  = f"{table_prefix}_scoring"
        self.table_metrics  = f"{table_prefix}_metrics"

    # ── conexión ───────────────────────────────────────────────────────────

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=API_BASE,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=_TIMEOUT,
            )
        return self._client

    def _safe_ingest(self, table: str, rows: List[dict]) -> bool:
        if not self.online:
            self._store.ingest(table, rows)
            return True
        try:
            self.client.post(f"/v1/tables/{table}", json={"rows": rows})
            return True
        except Exception:
            # fallback offline
            self._store.ingest(table, rows)
            return False

    def _safe_query(self, sql: str) -> List[dict]:
        if not self.online:
            return self._store.query(sql)
        try:
            resp = self.client.post("/v1/query", json={"sql": sql})
            resp.raise_for_status()
            return resp.json().get("rows", [])
        except Exception:
            return self._store.query(sql)

    # ── ingesta de decisiones ──────────────────────────────────────────────

    def log_decision(
        self,
        timestamp: str,
        idx: int,
        soc_kwh: float,
        soc_pct: float,
        cap_kwh: float,
        power_kw: float,
        price: float,
        co2: float,
        load_net: float,
        mode: str,
        horizon: int,
        scenarios: int,
        lp_time: float,
        elapsed_total: float,
    ) -> None:
        """Registra una decisión individual del agente."""
        row = {
            "ts": timestamp,
            "idx": idx,
            "soc_kwh": round(soc_kwh, 2),
            "soc_pct": round(soc_pct, 2),
            "cap_kwh": round(cap_kwh, 2),
            "power_kw": round(power_kw, 3),
            "price": round(price, 6),
            "co2": round(co2, 6),
            "load_net": round(load_net, 2),
            "mode": mode,
            "horizon": horizon,
            "scenarios": scenarios,
            "lp_time_ms": round(lp_time * 1000, 1),
            "elapsed_total_s": round(elapsed_total, 3),
            "run_id": self._run_id(),
        }
        self._safe_ingest(self.table_decisions, [row])

    def log_batch(self, decisions: List[dict]) -> None:
        """Ingesta masiva de decisiones (útil al final de la ejecución)."""
        self._safe_ingest(self.table_decisions, decisions)

    def log_scoring(self, total_score: float, total_cost: float, total_co2: float) -> None:
        """Registra el score final."""
        row = {
            "ts": datetime.datetime.utcnow().isoformat(),
            "total_score": round(total_score, 2),
            "total_cost": round(total_cost, 4),
            "total_co2": round(total_co2, 4),
            "run_id": self._run_id(),
        }
        self._safe_ingest(self.table_scoring, [row])

    def log_metrics(self, metrics: dict) -> None:
        """Registra métricas agregadas (p. ej. del profiling)."""
        row = {
            "ts": datetime.datetime.utcnow().isoformat(),
            "run_id": self._run_id(),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        }
        self._safe_ingest(self.table_metrics, [row])

    # ── consultas ──────────────────────────────────────────────────────────

    def get_decisions(self, limit: int = 100) -> List[dict]:
        """Últimas N decisiones."""
        return self._safe_query(
            f"SELECT * FROM {self.table_decisions} ORDER BY idx DESC LIMIT {limit}"
        )

    def get_average_lp_time(self) -> float:
        """Tiempo medio del LP en milisegundos."""
        rows = self._safe_query(
            f"SELECT AVG(lp_time_ms) as avg_ms FROM {self.table_decisions}"
        )
        if rows and "avg_ms" in rows[0]:
            return rows[0]["avg_ms"]
        return 0.0

    def get_power_distribution(self) -> List[dict]:
        """Distribución de la potencia: cuántas veces cargó, descargó, quieto."""
        if not self.online:
            decisions = self.get_decisions(limit=1000)
            charge = [d for d in decisions if isinstance(d, dict) and d.get("power_kw", 0) > 1]
            discharge = [d for d in decisions if isinstance(d, dict) and d.get("power_kw", 0) < -1]
            idle = [d for d in decisions if isinstance(d, dict) and -1 <= d.get("power_kw", 0) <= 1]
            return [
                {"action": "charge", "n": len(charge), "avg_kw": round(sum(d.get("power_kw", 0) for d in charge) / len(charge), 3) if charge else 0},
                {"action": "discharge", "n": len(discharge), "avg_kw": round(sum(d.get("power_kw", 0) for d in discharge) / len(discharge), 3) if discharge else 0},
                {"action": "idle", "n": len(idle), "avg_kw": round(sum(d.get("power_kw", 0) for d in idle) / len(idle), 3) if idle else 0},
            ]
        return self._safe_query(
            f"""
            SELECT
                CASE
                    WHEN power_kw > 1 THEN 'charge'
                    WHEN power_kw < -1 THEN 'discharge'
                    ELSE 'idle'
                END as action,
                COUNT(*) as n,
                ROUND(AVG(power_kw), 3) as avg_kw
            FROM {self.table_decisions}
            GROUP BY action
            ORDER BY n DESC
            """
        )

    def get_score(self) -> Optional[float]:
        """Score final del agente."""
        rows = self._safe_query(
            f"SELECT total_score FROM {self.table_scoring} ORDER BY ts DESC LIMIT 1"
        )
        if rows and "total_score" in rows[0]:
            return rows[0]["total_score"]
        return None

    # ── análisis embebido (el agente se consulta a sí mismo) ───────────────

    def analyze(self) -> dict:
        """Genera un análisis automático de las decisiones tomadas.

        El agente puede usar esta información para ajustar su comportamiento
        en llamadas futuras (p. ej. reducir la carga si detecta que el precio
        fue más alto de lo esperado).
        """
        decisions = self.get_decisions(limit=1000)
        if not decisions:
            return {"message": "Sin decisiones registradas"}

        dist = self.get_power_distribution()
        avg_lp = self.get_average_lp_time()
        scores = self._safe_query(
            f"SELECT total_score FROM {self.table_scoring}"
        )

        # Distribución de potencias en histograma
        import numpy as np
        powers = [d.get("power_kw", 0) for d in decisions if isinstance(d, dict)]
        if powers:
            hist, edges = np.histogram(powers, bins=20, density=True)
            histogram = [{"bin_low": round(edges[i], 2), "bin_high": round(edges[i + 1], 2), "density": round(float(h), 4)}
                         for i, h in enumerate(hist) if h > 0]
        else:
            histogram = []

        return {
            "n_decisions": len(decisions),
            "power_distribution": dist,
            "avg_lp_time_ms": avg_lp,
            "scores": scores,
            "power_histogram": histogram,
            "first_ts": decisions[-1].get("ts") if decisions else None,
            "last_ts": decisions[0].get("ts") if decisions else None,
        }

    # ── utilidades ─────────────────────────────────────────────────────────

    _run_id_val: str | None = None

    @classmethod
    def _run_id(cls) -> str:
        if cls._run_id_val is None:
            cls._run_id_val = str(uuid.uuid4())[:8]
        return cls._run_id_val

    def flush(self) -> None:
        """Cierra la conexión HTTP."""
        if self._client:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        self.flush()