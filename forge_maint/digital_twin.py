"""Digital-twin-style validation and safety rule checking.

This is a lightweight surrogate of the digital twin validation module. In real
systems it can call finite element simulation, process twins, equipment twins,
or a plant safety interlock service.
"""

from __future__ import annotations

from typing import List

from .schemas import FusionResult, MaintenanceScenario, ValidationReport


class DigitalTwinValidator:
    def __init__(self):
        self.rules = {
            "temperature_c_max": 85.0,
            "pressure_mpa_max": 28.0,
            "load_ton_max": 720.0,
            "unknown_score_review_threshold": 0.60,
            "low_health_review_threshold": 0.45,
        }

    def validate(self, scenario: MaintenanceScenario, fusion: FusionResult) -> ValidationReport:
        conflicts: List[str] = []
        violated: List[str] = []
        suggestions: List[str] = []
        op = scenario.operating_condition
        temp = float(op.get("temperature_c", 25.0))
        pressure = float(op.get("pressure_mpa", 0.0))
        load = float(op.get("load_ton", 0.0))

        if temp > self.rules["temperature_c_max"]:
            violated.append("temperature_c_max")
            conflicts.append(f"Current temperature {temp:.1f}°C exceeds safety rule limit.")
            suggestions.append("Reduce load or pause operation for thermal inspection.")
        if pressure > self.rules["pressure_mpa_max"]:
            violated.append("pressure_mpa_max")
            conflicts.append(f"Current pressure {pressure:.1f} MPa exceeds hydraulic safety limit.")
            suggestions.append("Check hydraulic circuit, relief valve and pressure sensor calibration.")
        if load > self.rules["load_ton_max"]:
            violated.append("load_ton_max")
            conflicts.append(f"Current load {load:.1f} ton exceeds allowed envelope.")
            suggestions.append("Verify billet size, die alignment and overload protection settings.")
        if fusion.uncertainty > self.rules["unknown_score_review_threshold"]:
            violated.append("uncertainty_review")
            suggestions.append("Trigger human review because unknown-fault uncertainty is high.")
        if fusion.health_score < self.rules["low_health_review_threshold"]:
            violated.append("low_health_review")
            suggestions.append("Schedule immediate inspection for high-risk degradation.")

        high_risk_action = any(
            token in " ".join(fusion.recommended_actions).lower()
            for token in ["stop", "shutdown", "停机", "立即停机", "参数修改"]
        )
        require_human = bool(violated) or scenario.risk_level.lower() in {"high", "critical"} or high_risk_action
        passed = not conflicts
        if not suggestions and passed:
            suggestions.append("No digital-twin safety conflict detected; continue monitoring.")
        risk_level = "critical" if conflicts else ("high" if require_human else scenario.risk_level)
        return ValidationReport(
            passed=passed,
            risk_level=risk_level,
            conflicts=conflicts,
            violated_rules=violated,
            suggestions=suggestions,
            require_human_review=require_human,
        )
