from pathlib import Path

from forge_maint.controller import MultiAgentMaintenanceSystem
from forge_maint.utils import load_json


def test_high_risk_demo_runs():
    root = Path(__file__).resolve().parents[1]
    system = MultiAgentMaintenanceSystem(
        model_profiles_path=root / "config" / "model_profiles.json",
        knowledge_path=root / "knowledge" / "forging_knowledge.json",
        feedback_store_path=root / "outputs" / "feedback_records_test.json",
        output_dir=root / "outputs",
    )
    raw = load_json(root / "examples" / "sample_event_high_risk.json")
    decision = system.run_closed_loop_once(raw, save_result=False)
    assert decision.fusion.model_outputs
    assert decision.final_actions
    assert decision.evidence_chain["model_versions"]
