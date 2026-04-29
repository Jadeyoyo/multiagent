"""PHM model pool and demo model implementations.

The models included here are intentionally simple deterministic placeholders.
They implement the same interface that real bearing/press/forging PHM models
should implement, so you can replace them with CNN/Transformer/DA/UniDA models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional
import time

from .schemas import MaintenanceScenario, ModelOutput, PHMModelProfile, SignalSummary
from .utils import clamp, load_json, load_signal_csv, summarize_signal


class BasePHMModel:
    def __init__(self, profile: PHMModelProfile):
        self.profile = profile

    def predict(self, scenario: MaintenanceScenario, signal: List[float]) -> ModelOutput:
        raise NotImplementedError

    def _output(
        self,
        start: float,
        fault_label: str,
        health_score: float,
        confidence: float,
        unknown_score: float,
        evidence: dict,
        warnings: Optional[List[str]] = None,
    ) -> ModelOutput:
        elapsed = int((time.time() - start) * 1000)
        return ModelOutput(
            model_id=self.profile.model_id,
            model_name=self.profile.name,
            version=self.profile.version,
            fault_label=fault_label,
            health_score=round(clamp(health_score), 4),
            confidence=round(clamp(confidence), 4),
            unknown_score=round(clamp(unknown_score), 4),
            latency_ms=max(elapsed, self.profile.latency_ms),
            evidence=evidence,
            warnings=warnings or [],
        )


class StatisticalThresholdModel(BasePHMModel):
    """Fast baseline model for low-risk vibration anomaly screening."""

    def predict(self, scenario: MaintenanceScenario, signal: List[float]) -> ModelOutput:
        start = time.time()
        summary = summarize_signal(signal)
        temp = float(scenario.operating_condition.get("temperature_c", 25.0))
        load_ton = float(scenario.operating_condition.get("load_ton", 0.0))
        rms_alert = summary.rms > 1.20
        impulsive_alert = summary.crest_factor > 4.5 or summary.kurtosis_like > 5.0
        thermal_alert = temp > 75.0
        load_alert = load_ton > 650.0
        if thermal_alert and rms_alert:
            label = "thermal_mechanical_coupled_abnormality"
            confidence = 0.76
            health = 0.38
        elif impulsive_alert:
            label = "impact_or_bearing_defect_suspected"
            confidence = 0.70
            health = 0.45
        elif rms_alert or load_alert:
            label = "load_related_vibration_abnormality"
            confidence = 0.64
            health = 0.58
        else:
            label = "normal_or_mild_degradation"
            confidence = 0.80
            health = 0.86
        unknown = 0.20 + 0.35 * float(scenario.unknown_fault_risk)
        return self._output(
            start,
            label,
            health,
            confidence,
            unknown,
            evidence={"signal_summary": summary.__dict__, "rules": ["rms", "crest_factor", "temperature", "load"]},
        )


class UniversalDomainAdaptationDemoModel(BasePHMModel):
    """Demo model representing UDA/UniDA-style PHM model with unknown detection."""

    def predict(self, scenario: MaintenanceScenario, signal: List[float]) -> ModelOutput:
        start = time.time()
        summary = summarize_signal(signal)
        drift_score = clamp(abs(float(scenario.operating_condition.get("speed_spm", 60.0)) - 60.0) / 80.0)
        signal_outlier = clamp((summary.rms - 0.9) / 1.4 + max(summary.kurtosis_like - 3.0, 0.0) / 8.0)
        unknown = clamp(0.25 * scenario.unknown_fault_risk + 0.40 * signal_outlier + 0.25 * drift_score)
        if unknown > 0.62:
            label = "unknown_or_open_set_fault"
            confidence = 0.66 + 0.20 * unknown
            health = 0.32
        elif summary.rms > 1.15 and summary.kurtosis_like > 4.0:
            label = "bearing_or_guiding_system_fault"
            confidence = 0.78
            health = 0.41
        elif float(scenario.operating_condition.get("pressure_mpa", 0.0)) > 24.0:
            label = "hydraulic_pressure_instability"
            confidence = 0.72
            health = 0.52
        else:
            label = "known_condition_normal_or_minor_fault"
            confidence = 0.74
            health = 0.78
        return self._output(
            start,
            label,
            health,
            confidence,
            unknown,
            evidence={
                "signal_summary": summary.__dict__,
                "domain_shift_score": round(drift_score, 4),
                "signal_outlier_score": round(signal_outlier, 4),
                "label_setting": scenario.label_status,
            },
        )


class ProcessRuleHybridModel(BasePHMModel):
    """Hybrid model using process metadata and signal features."""

    def predict(self, scenario: MaintenanceScenario, signal: List[float]) -> ModelOutput:
        start = time.time()
        summary = summarize_signal(signal)
        pressure = float(scenario.operating_condition.get("pressure_mpa", 0.0))
        temp = float(scenario.operating_condition.get("temperature_c", 25.0))
        speed = float(scenario.operating_condition.get("speed_spm", 60.0))
        warnings = []
        if summary.missing_ratio > 0.05:
            warnings.append("Signal missing ratio is high; model confidence reduced.")
        process_severity = clamp((pressure - 18.0) / 12.0 + (temp - 55.0) / 60.0 + abs(speed - 60.0) / 100.0)
        if process_severity > 0.72 and summary.rms > 1.0:
            label = "process_parameter_induced_abnormality"
            health = 0.36
            conf = 0.73
        elif pressure > 25.0:
            label = "hydraulic_overload_risk"
            health = 0.50
            conf = 0.69
        elif temp > 80.0:
            label = "thermal_overload_risk"
            health = 0.48
            conf = 0.70
        else:
            label = "process_condition_acceptable"
            health = 0.82
            conf = 0.68
        if warnings:
            conf *= 0.75
        unknown = clamp(0.18 + 0.20 * scenario.unknown_fault_risk + 0.20 * process_severity)
        return self._output(
            start,
            label,
            health,
            conf,
            unknown,
            evidence={
                "signal_summary": summary.__dict__,
                "process_severity": round(process_severity, 4),
                "checked_conditions": {"pressure_mpa": pressure, "temperature_c": temp, "speed_spm": speed},
            },
            warnings=warnings,
        )


MODEL_CLASS_REGISTRY = {
    "statistical_threshold": StatisticalThresholdModel,
    "unida_demo": UniversalDomainAdaptationDemoModel,
    "process_rule_hybrid": ProcessRuleHybridModel,
}


class ModelPool:
    def __init__(self, profiles_path: str | Path):
        self.profiles_path = Path(profiles_path)
        raw_profiles = load_json(self.profiles_path)
        self.profiles: Dict[str, PHMModelProfile] = {}
        self.models: Dict[str, BasePHMModel] = {}
        for item in raw_profiles:
            profile = PHMModelProfile(**item["profile"])
            model_type = item["model_type"]
            if model_type not in MODEL_CLASS_REGISTRY:
                raise ValueError(f"Unknown model_type: {model_type}")
            self.profiles[profile.model_id] = profile
            self.models[profile.model_id] = MODEL_CLASS_REGISTRY[model_type](profile)

    def list_profiles(self) -> List[PHMModelProfile]:
        return list(self.profiles.values())

    def get_profile(self, model_id: str) -> PHMModelProfile:
        return self.profiles[model_id]

    def run_model(self, model_id: str, scenario: MaintenanceScenario) -> ModelOutput:
        signal = []
        if scenario.available_signal_path:
            signal_path = Path(scenario.available_signal_path)
            if not signal_path.is_absolute():
                # Resolve from project root if a relative path was passed in config/examples.
                signal_path = Path.cwd() / signal_path
            if signal_path.exists():
                signal = load_signal_csv(signal_path)
        output = self.models[model_id].predict(scenario, signal)
        self.profiles[model_id].usage_count += 1
        return output

    def run_models(self, model_ids: Iterable[str], scenario: MaintenanceScenario) -> List[ModelOutput]:
        outputs = []
        for mid in model_ids:
            if mid not in self.models:
                continue
            outputs.append(self.run_model(mid, scenario))
        return outputs
