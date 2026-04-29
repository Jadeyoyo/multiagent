"""Validation agent."""

from __future__ import annotations

from .base import BaseAgent
from ..digital_twin import DigitalTwinValidator
from ..schemas import FusionResult, MaintenanceScenario, ValidationReport


class ValidationAgent(BaseAgent):
    def __init__(self, validator: DigitalTwinValidator):
        super().__init__("ValidationAgent")
        self.validator = validator

    def validate(self, scenario: MaintenanceScenario, fusion: FusionResult) -> ValidationReport:
        report = self.validator.validate(scenario, fusion)
        self.log("Digital twin validation completed.", passed=report.passed, risk_level=report.risk_level)
        return report
