"""Continuous updating agent."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .base import BaseAgent
from ..schemas import FeedbackRecord, MaintenanceDecision, utc_now_iso
from ..utils import load_json, save_json


class UpdateAgent(BaseAgent):
    def __init__(self, feedback_store_path: str | Path):
        super().__init__("UpdateAgent")
        self.feedback_store_path = Path(feedback_store_path)
        self.feedback_store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.feedback_store_path.exists():
            save_json(self.feedback_store_path, [])

    def record_feedback(
        self,
        decision: MaintenanceDecision,
        operator_feedback: str,
        actual_fault_label: Optional[str] = None,
        maintenance_result: Optional[str] = None,
        followup_health_status: Optional[str] = None,
    ) -> Dict[str, object]:
        record = FeedbackRecord(
            decision_id=decision.decision_id,
            timestamp=utc_now_iso(),
            operator_feedback=operator_feedback,
            actual_fault_label=actual_fault_label,
            maintenance_result=maintenance_result,
            followup_health_status=followup_health_status,
            notes=[],
        )
        records: List[dict] = load_json(self.feedback_store_path)
        records.append(record.__dict__)
        save_json(self.feedback_store_path, records)
        triggers = self._derive_update_triggers(decision, record)
        self.log("Feedback recorded and update triggers derived.", trigger_count=len(triggers))
        return {"feedback_record": record.__dict__, "update_triggers": triggers}

    def _derive_update_triggers(self, decision: MaintenanceDecision, record: FeedbackRecord) -> List[str]:
        triggers: List[str] = []
        if "否决" in record.operator_feedback or "reject" in record.operator_feedback.lower():
            triggers.append("update_knowledge_rules_or_thresholds_due_to_operator_rejection")
        if record.actual_fault_label and record.actual_fault_label != decision.fusion.final_label:
            triggers.append("add_labeled_sample_and_retrain_or_calibrate_model")
        if decision.fusion.uncertainty > 0.65:
            triggers.append("collect_more_samples_for_unknown_fault_or_domain_shift")
        if decision.scenario.data_quality_score < 0.60:
            triggers.append("check_sensor_quality_and_data_pipeline")
        if decision.validation.violated_rules:
            triggers.append("update_safety_rule_statistics_and_case_library")
        if not triggers:
            triggers.append("no_immediate_update_required_keep_monitoring")
        return triggers
