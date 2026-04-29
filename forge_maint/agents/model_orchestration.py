"""Model orchestration agent."""

from __future__ import annotations

from typing import List, Tuple

from .base import BaseAgent
from ..model_pool import ModelPool
from ..schemas import MaintenanceScenario, ModelCallPlan, PHMModelProfile


class ModelOrchestrationAgent(BaseAgent):
    def __init__(self, model_pool: ModelPool):
        super().__init__("ModelOrchestrationAgent")
        self.model_pool = model_pool

    def _score_profile(self, scenario: MaintenanceScenario, profile: PHMModelProfile) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []
        if scenario.task_type in profile.supported_tasks:
            score += 0.25
            reasons.append(f"supports task {scenario.task_type}")
        modality_overlap = set(scenario.data_modalities) & set(profile.supported_modalities)
        if modality_overlap:
            score += 0.20 + 0.05 * len(modality_overlap)
            reasons.append(f"matches modalities {sorted(modality_overlap)}")
        if scenario.label_status in profile.label_settings:
            score += 0.20
            reasons.append(f"matches label setting {scenario.label_status}")
        elif scenario.label_status in {"open_set", "opda", "universal_da"} and profile.supports_unknown_detection:
            score += 0.18
            reasons.append("supports unknown-fault detection")
        score += 0.20 * profile.reliability_score
        if scenario.unknown_fault_risk > 0.55 and profile.supports_unknown_detection:
            score += 0.15
            reasons.append("preferred due to high unknown-fault risk")
        if scenario.real_time_requirement == "real_time" and profile.latency_ms < 100:
            score += 0.10
            reasons.append("low-latency model")
        if profile.recent_error_count > 0:
            score -= min(0.15, 0.03 * profile.recent_error_count)
            reasons.append("penalized by recent errors")
        return score, reasons

    def make_plan(self, scenario: MaintenanceScenario) -> ModelCallPlan:
        scored = []
        for profile in self.model_pool.list_profiles():
            score, reasons = self._score_profile(scenario, profile)
            if score > 0.35:
                scored.append((score, profile.model_id, reasons))
        scored.sort(key=lambda x: x[0], reverse=True)
        high_risk = scenario.risk_level in {"high", "critical"} or scenario.unknown_fault_risk > 0.60
        if not scored:
            return ModelCallPlan(
                selected_model_ids=[],
                strategy="no_model_available",
                reasons=["No model profile satisfied the current scenario. Supplement data or register a new model."],
                require_human_review=True,
                require_digital_twin_validation=True,
            )
        if high_risk:
            selected = [mid for _, mid, _ in scored[: min(3, len(scored))]]
            strategy = "multi_model_consistency_check"
            require_dt = True
            require_hr = True
        else:
            selected = [scored[0][1]]
            strategy = "single_best_model"
            require_dt = scenario.risk_level == "medium"
            require_hr = False
        reasons = []
        for score, mid, rs in scored[: len(selected)]:
            reasons.append(f"{mid}: score={score:.3f}; " + "; ".join(rs))
        plan = ModelCallPlan(
            selected_model_ids=selected,
            strategy=strategy,
            reasons=reasons,
            require_human_review=require_hr,
            require_digital_twin_validation=require_dt,
        )
        self.log("Model call plan generated.", strategy=strategy, selected=selected)
        return plan
