"""Lightweight domain knowledge retrieval module.

In a production deployment, this component can be replaced by a vector database
or RAG system. Here it uses deterministic keyword retrieval so the prototype is
fully runnable without network access or paid APIs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .schemas import KnowledgeEvidence, MaintenanceScenario
from .utils import keyword_terms, load_json


class KnowledgeBase:
    def __init__(self, articles_path: str | Path):
        self.articles_path = Path(articles_path)
        self.articles = load_json(self.articles_path)

    def retrieve(
        self,
        scenario: MaintenanceScenario,
        query_terms: Iterable[str],
        top_k: int = 5,
    ) -> List[KnowledgeEvidence]:
        scenario_text = " ".join(
            [
                scenario.equipment_type,
                scenario.task_type,
                scenario.process_stage,
                scenario.risk_level,
                scenario.label_status,
                " ".join(scenario.data_modalities),
                " ".join(str(v) for v in scenario.operating_condition.values()),
            ]
        )
        q_terms = set(keyword_terms(scenario_text)) | {str(t).lower() for t in query_terms if str(t).strip()}
        scored = []
        for item in self.articles:
            content_terms = keyword_terms(
                " ".join(
                    [
                        item.get("title", ""),
                        item.get("content", ""),
                        " ".join(item.get("keywords", [])),
                        " ".join(item.get("fault_modes", [])),
                        " ".join(item.get("applicable_equipment", [])),
                    ]
                )
            )
            matched = sorted(q_terms & content_terms)
            # Additional score boost for matching equipment/task/modality fields.
            field_boost = 0.0
            if scenario.equipment_type.lower() in [x.lower() for x in item.get("applicable_equipment", [])]:
                field_boost += 0.25
            if scenario.task_type.lower() in [x.lower() for x in item.get("tasks", [])]:
                field_boost += 0.20
            if set(m.lower() for m in scenario.data_modalities) & set(x.lower() for x in item.get("modalities", [])):
                field_boost += 0.15
            score = len(matched) / max(len(q_terms), 1) + field_boost
            if score > 0:
                scored.append((score, matched, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        evidence = []
        for score, matched, item in scored[:top_k]:
            evidence.append(
                KnowledgeEvidence(
                    source_id=item.get("source_id", "unknown"),
                    title=item.get("title", "Untitled"),
                    source_type=item.get("source_type", "knowledge"),
                    relevance_score=round(score, 4),
                    matched_terms=matched[:20],
                    content=item.get("content", ""),
                    recommended_actions=item.get("recommended_actions", []),
                    safety_notes=item.get("safety_notes", []),
                )
            )
        return evidence
