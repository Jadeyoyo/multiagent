"""Result fusion and explanation agent."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .base import BaseAgent
from ..model_pool import ModelPool
from ..schemas import FusionResult, KnowledgeEvidence, MaintenanceScenario, ModelOutput
from ..utils import clamp


class FusionExplanationAgent(BaseAgent):
    def __init__(self, model_pool: ModelPool):
        super().__init__("FusionExplanationAgent")
        self.model_pool = model_pool

    def fuse(
        self,
        scenario: MaintenanceScenario,
        outputs: List[ModelOutput],
        evidence: List[KnowledgeEvidence],
        pre_require_human_review: bool = False,
    ) -> FusionResult:
        if not outputs:
            return FusionResult(
                final_label="no_model_output",
                health_score=0.0,
                confidence=0.0,
                uncertainty=1.0,
                explanation="No PHM model was called or all model calls failed. The system cannot produce a reliable diagnostic conclusion.",
                conflicts=["No model output available."],
                model_outputs=[],
                knowledge_evidence=evidence,
                recommended_actions=["Supplement sensor data and request expert review."],
                require_human_review=True,
            )

        label_scores: Dict[str, float] = defaultdict(float)
        health_num = 0.0
        health_den = 0.0
        unknown_scores = []
        conflicts = []
        for out in outputs:
            profile = self.model_pool.get_profile(out.model_id)
            weight = max(0.05, out.confidence * profile.reliability_score)
            if out.unknown_score > 0.65 or "unknown" in out.fault_label:
                label_scores["unknown_or_open_set_fault"] += weight * 1.15
            label_scores[out.fault_label] += weight
            health_num += out.health_score * weight
            health_den += weight
            unknown_scores.append(out.unknown_score)
        final_label = max(label_scores.items(), key=lambda x: x[1])[0]
        health = health_num / max(health_den, 1e-8)
        confidence = clamp(max(label_scores.values()) / max(sum(label_scores.values()), 1e-8))
        uncertainty = clamp(sum(unknown_scores) / len(unknown_scores) + (1.0 - confidence) * 0.35)

        labels = {o.fault_label for o in outputs}
        if len(labels) > 1:
            conflicts.append("Different PHM models produced inconsistent labels: " + ", ".join(sorted(labels)))
        if any(o.warnings for o in outputs):
            conflicts.append("At least one PHM model reported data/model warning.")
        if scenario.data_quality_score < 0.60:
            conflicts.append("Low data quality may reduce diagnostic reliability.")

        actions = []
        for item in evidence:
            actions.extend(item.recommended_actions)
        if not actions:
            if health < 0.45 or uncertainty > 0.65:
                actions = ["Trigger manual inspection and collect additional vibration, temperature and pressure data."]
            elif health < 0.65:
                actions = ["Increase monitoring frequency and schedule preventive inspection."]
            else:
                actions = ["Continue operation under routine monitoring."]
        # Deduplicate while keeping order.
        dedup_actions = []
        for a in actions:
            if a not in dedup_actions:
                dedup_actions.append(a)

        evidence_titles = "; ".join([f"{e.title}({e.source_type})" for e in evidence[:3]]) or "no knowledge evidence"
        explanation = (
            f"The fused conclusion is '{final_label}'. The average health score is {health:.2f}, "
            f"the fused confidence is {confidence:.2f}, and the uncertainty is {uncertainty:.2f}. "
            f"The decision used {len(outputs)} PHM model output(s) and retrieved evidence from {evidence_titles}."
        )
        if conflicts:
            explanation += " Conflicts were detected and should be reviewed: " + " | ".join(conflicts)
        require_hr = pre_require_human_review or scenario.risk_level in {"high", "critical"} or uncertainty > 0.60 or bool(conflicts)
        result = FusionResult(
            final_label=final_label,
            health_score=round(health, 4),
            confidence=round(confidence, 4),
            uncertainty=round(uncertainty, 4),
            explanation=explanation,
            conflicts=conflicts,
            model_outputs=outputs,
            knowledge_evidence=evidence,
            recommended_actions=dedup_actions[:8],
            require_human_review=require_hr,
        )
        self.log("Model outputs fused and explanation generated.", final_label=final_label, uncertainty=result.uncertainty)
        return result
