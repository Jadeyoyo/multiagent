"""Knowledge retrieval agent."""

from __future__ import annotations

from typing import Iterable, List

from .base import BaseAgent
from ..knowledge_base import KnowledgeBase
from ..schemas import KnowledgeEvidence, MaintenanceScenario, ModelOutput


class KnowledgeRetrievalAgent(BaseAgent):
    def __init__(self, knowledge_base: KnowledgeBase):
        super().__init__("KnowledgeRetrievalAgent")
        self.knowledge_base = knowledge_base

    def retrieve_for_scenario(
        self,
        scenario: MaintenanceScenario,
        model_outputs: Iterable[ModelOutput] | None = None,
        top_k: int = 5,
    ) -> List[KnowledgeEvidence]:
        query_terms = [scenario.equipment_type, scenario.task_type, scenario.process_stage, scenario.label_status]
        for k, v in scenario.operating_condition.items():
            query_terms.extend([str(k), str(v)])
        if model_outputs:
            for out in model_outputs:
                query_terms.extend([out.fault_label, out.model_name])
                if out.unknown_score > 0.60:
                    query_terms.extend(["unknown", "open_set", "manual_review"])
        evidence = self.knowledge_base.retrieve(scenario, query_terms=query_terms, top_k=top_k)
        self.log("Knowledge evidence retrieved.", evidence_count=len(evidence))
        return evidence
