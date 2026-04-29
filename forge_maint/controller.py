"""Top-level multi-agent closed-loop controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import uuid

from .agents.scenario_understanding import ScenarioUnderstandingAgent
from .agents.model_orchestration import ModelOrchestrationAgent
from .agents.knowledge_retrieval import KnowledgeRetrievalAgent
from .agents.fusion_explanation import FusionExplanationAgent
from .agents.validation import ValidationAgent
from .agents.update import UpdateAgent
from .digital_twin import DigitalTwinValidator
from .knowledge_base import KnowledgeBase
from .model_pool import ModelPool
from .schemas import MaintenanceDecision
from .utils import save_json


class MultiAgentMaintenanceSystem:
    def __init__(
        self,
        model_profiles_path: str | Path,
        knowledge_path: str | Path,
        feedback_store_path: str | Path,
        output_dir: str | Path = "outputs",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_pool = ModelPool(model_profiles_path)
        self.knowledge_base = KnowledgeBase(knowledge_path)
        self.scenario_agent = ScenarioUnderstandingAgent()
        self.orchestration_agent = ModelOrchestrationAgent(self.model_pool)
        self.knowledge_agent = KnowledgeRetrievalAgent(self.knowledge_base)
        self.fusion_agent = FusionExplanationAgent(self.model_pool)
        self.validation_agent = ValidationAgent(DigitalTwinValidator())
        self.update_agent = UpdateAgent(feedback_store_path)

    def run_closed_loop_once(self, raw_event: Dict[str, Any], save_result: bool = True) -> MaintenanceDecision:
        scenario = self.scenario_agent.understand(raw_event)
        plan = self.orchestration_agent.make_plan(scenario)
        model_outputs = self.model_pool.run_models(plan.selected_model_ids, scenario)
        evidence = self.knowledge_agent.retrieve_for_scenario(scenario, model_outputs=model_outputs)
        fusion = self.fusion_agent.fuse(
            scenario,
            outputs=model_outputs,
            evidence=evidence,
            pre_require_human_review=plan.require_human_review,
        )
        validation = self.validation_agent.validate(scenario, fusion)
        final_actions = list(fusion.recommended_actions)
        for s in validation.suggestions:
            if s not in final_actions:
                final_actions.append(s)
        if validation.require_human_review or fusion.require_human_review:
            final_actions.insert(0, "人工复核：高风险、不确定或存在规则冲突，智能体仅输出建议，不直接执行停机或参数修改。")

        evidence_chain = {
            "model_plan": plan.__dict__,
            "model_versions": [
                {
                    "model_id": o.model_id,
                    "model_name": o.model_name,
                    "version": o.version,
                    "key_output": {
                        "fault_label": o.fault_label,
                        "health_score": o.health_score,
                        "confidence": o.confidence,
                        "unknown_score": o.unknown_score,
                    },
                }
                for o in model_outputs
            ],
            "knowledge_sources": [
                {
                    "source_id": e.source_id,
                    "title": e.title,
                    "source_type": e.source_type,
                    "relevance_score": e.relevance_score,
                }
                for e in evidence
            ],
            "validation": validation.__dict__,
            "agent_traces": {
                "scenario_agent": self.scenario_agent.trace.__dict__,
                "orchestration_agent": self.orchestration_agent.trace.__dict__,
                "knowledge_agent": self.knowledge_agent.trace.__dict__,
                "fusion_agent": self.fusion_agent.trace.__dict__,
                "validation_agent": self.validation_agent.trace.__dict__,
            },
        }
        decision = MaintenanceDecision(
            decision_id=f"D-{uuid.uuid4().hex[:10]}",
            scenario=scenario,
            plan=plan,
            fusion=fusion,
            validation=validation,
            final_actions=final_actions,
            evidence_chain=evidence_chain,
        )
        if save_result:
            save_json(self.output_dir / f"{decision.decision_id}.json", decision)
        return decision

    def record_feedback(self, decision: MaintenanceDecision, operator_feedback: str, **kwargs: Optional[str]) -> Dict[str, object]:
        return self.update_agent.record_feedback(decision, operator_feedback=operator_feedback, **kwargs)
