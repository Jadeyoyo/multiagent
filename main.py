"""Command-line entry point for the forging multi-agent maintenance prototype.

Run examples:
  python main.py --event examples/sample_event_high_risk.json
  python main.py --event examples/sample_event_low_risk.json --feedback "专家确认需要检查导轨"
"""

from __future__ import annotations

from pathlib import Path
import argparse

from forge_maint.controller import MultiAgentMaintenanceSystem
from forge_maint.schemas import to_pretty_json
from forge_maint.utils import load_json


PROJECT_ROOT = Path(__file__).resolve().parent


def build_system() -> MultiAgentMaintenanceSystem:
    return MultiAgentMaintenanceSystem(
        model_profiles_path=PROJECT_ROOT / "config" / "model_profiles.json",
        knowledge_path=PROJECT_ROOT / "knowledge" / "forging_knowledge.json",
        feedback_store_path=PROJECT_ROOT / "outputs" / "feedback_records.json",
        output_dir=PROJECT_ROOT / "outputs",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Forging intelligent maintenance multi-agent prototype")
    parser.add_argument("--event", default="examples/sample_event_high_risk.json", help="Path to raw maintenance event JSON")
    parser.add_argument("--feedback", default=None, help="Optional operator feedback to record after decision")
    parser.add_argument("--actual-fault-label", default=None, help="Optional confirmed actual fault label")
    parser.add_argument("--maintenance-result", default=None, help="Optional maintenance result")
    parser.add_argument("--followup-health-status", default=None, help="Optional follow-up health status")
    args = parser.parse_args()

    event_path = Path(args.event)
    if not event_path.is_absolute():
        event_path = PROJECT_ROOT / event_path
    raw_event = load_json(event_path)

    system = build_system()
    decision = system.run_closed_loop_once(raw_event, save_result=True)
    print("\n=== Multi-agent maintenance decision ===")
    print(to_pretty_json(decision))
    print(f"\nSaved decision JSON to: {PROJECT_ROOT / 'outputs' / (decision.decision_id + '.json')}")

    if args.feedback:
        feedback_result = system.record_feedback(
            decision,
            operator_feedback=args.feedback,
            actual_fault_label=args.actual_fault_label,
            maintenance_result=args.maintenance_result,
            followup_health_status=args.followup_health_status,
        )
        print("\n=== Feedback update result ===")
        print(to_pretty_json(feedback_result))


if __name__ == "__main__":
    main()
