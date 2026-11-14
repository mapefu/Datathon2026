"""
agent.py
Agent de control amb limitació conservadora per evitar infraccions físiques.

Límits segons les especificacions del repte:
- Capacitat: 2.000 kWh
- Potència: +/- 500 kW
- Rendiment round-trip: 90 %
- Consigna positiva: càrrega
- Consigna negativa: descàrrega
"""

import math


class BatteryAgent:
    def __init__(self):
        self.llindar_preu_carrega = 0.08
        self.llindar_preu_descarrega = 0.16
        self.potencia_nominal_kw = 500.0
        self.capacitat_kwh = 2000.0

        # Marge conservador addicional per evitar arribar exactament als límits.
        self.marge_kwh = 0.5

    @staticmethod
    def _numero_valid(valor, per_defecte=0.0):
        try:
            x = float(valor)
            return x if math.isfinite(x) else per_defecte
        except (TypeError, ValueError):
            return per_defecte

    def act(self, obs: dict) -> float:
        # Si falta la dada essencial del SOC, no arrisquem cap moviment.
        soc = self._numero_valid(obs.get("soc_kwh"), -1.0)
        if soc < 0.0:
            return 0.0

        # Limitem també un SOC fora de rang per protegir el càlcul.
        soc = min(self.capacitat_kwh, max(0.0, soc))

        hora = int(self._numero_valid(obs.get("hour"), 0))
        preu = self._numero_valid(obs.get("preu_actual"), 0.0)
        co2 = self._numero_valid(obs.get("co2_actual"), 0.0)
        soc_pct = self._numero_valid(
            obs.get("soc_pct"),
            soc / self.capacitat_kwh
        )

        # Política de referència: només determina la consigna desitjada.
        consigna = 0.0

        if 1 <= hora <= 6 and preu < self.llindar_preu_carrega and soc_pct < 0.85:
            consigna = 350.0
        elif 11 <= hora <= 15 and preu < 0.10 and soc_pct < 0.95:
            consigna = 250.0
        elif 18 <= hora <= 22 and (
            preu > self.llindar_preu_descarrega or co2 > 0.10
        ) and soc_pct > 0.15:
            consigna = -450.0

        # Seguretat física conservadora.
        # Es fa servir eta = sqrt(0.90), tal com indica el document del repte.
        eta = math.sqrt(0.90)
        marge = self.marge_kwh

        espai_util = max(0.0, self.capacitat_kwh - soc - marge)
        energia_util = max(0.0, soc - marge)

        max_carrega = min(
            self.potencia_nominal_kw,
            max(0.0, espai_util / eta)
        )
        max_descarrega = min(
            self.potencia_nominal_kw,
            max(0.0, energia_util * eta)
        )

        # Si el SOC és als extrems, quedar-se en repòs és més segur.
        consigna_segura = max(-max_descarrega, min(max_carrega, consigna))

        # Última protecció: retorn finit i dins del límit de l'inversor.
        if not math.isfinite(consigna_segura):
            return 0.0

        return float(max(-500.0, min(500.0, consigna_segura)))
