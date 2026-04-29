"""Scenario understanding agent."""

from __future__ import annotations

from typing import Any, Dict
import uuid

from .base import BaseAgent
from ..schemas import MaintenanceScenario, utc_now_iso
from ..utils import clamp


class ScenarioUnderstandingAgent(BaseAgent):
    """Converts raw maintenance events into structured scenario S_t."""

    def __init__(self):
        super().__init__("ScenarioUnderstandingAgent")

    def understand(self, raw_event: Dict[str, Any]) -> MaintenanceScenario:
        op = raw_event.get("operating_condition", {}) or {}
        temp = float(op.get("temperature_c", 25.0))
        pressure = float(op.get("pressure_mpa", 0.0))
        load = float(op.get("load_ton", 0.0))
        speed = float(op.get("speed_spm", 60.0))
        data_quality = float(raw_event.get("data_quality_score", 0.85))
        unknown_fault_risk = float(raw_event.get("unknown_fault_risk", 0.30))

        severity = clamp((temp - 60.0) / 40.0) * 0.30 + clamp((pressure - 20.0) / 12.0) * 0.30 + clamp((load - 550.0) / 250.0) * 0.25 + clamp(abs(speed - 60.0) / 80.0) * 0.15
        severity = clamp(severity + 0.25 * unknown_fault_risk + 0.20 * (1 - data_quality))
        if severity > 0.78:
            risk = "critical"
        elif severity > 0.55:
            risk = "high"
        elif severity > 0.32:
            risk = "medium"
        else:
            risk = "low"

        notes = []
        if data_quality < 0.60:
            notes.append("Data quality is low; diagnosis should be conservative.")
        if unknown_fault_risk > 0.60:
            notes.append("Unknown fault risk is high; open-set or UniDA model should be considered.")
        if temp > 80 or pressure > 26 or load > 700:
            notes.append("Operating condition approaches or exceeds safe envelope.")

        scenario = MaintenanceScenario(
            scenario_id=raw_event.get("scenario_id", f"S-{uuid.uuid4().hex[:8]}"),
            timestamp=raw_event.get("timestamp", utc_now_iso()),
            equipment_id=raw_event.get("equipment_id", "unknown_equipment"),
            equipment_type=raw_event.get("equipment_type", "forging_press"),
            task_type=raw_event.get("task_type", "fault_diagnosis"),
            process_stage=raw_event.get("process_stage", "unknown_stage"),
            operating_condition=op,
            data_modalities=raw_event.get("data_modalities", ["vibration"]),
            data_quality_score=round(clamp(data_quality), 4),
            label_status=raw_event.get("label_status", "open_set"),
            unknown_fault_risk=round(clamp(unknown_fault_risk), 4),
            real_time_requirement=raw_event.get("real_time_requirement", "near_real_time"),
            risk_level=risk,
            available_signal_path=raw_event.get("available_signal_path"),
            constraints=raw_event.get("constraints", {}),
            notes=notes,
        )
        self.log("Structured maintenance scenario generated.", scenario_id=scenario.scenario_id, risk_level=risk)
        return scenario
