"""SSV4D: sparse shadow verification for LLM determinism."""

from ssv4d.client import SGLangShadowClient
from ssv4d.perturbation import PerturbationPlan, make_perturbation_plans
from ssv4d.policy import (
    CompositeRiskGate,
    DualThresholdGate,
    calibrate_and_evaluate,
    calibrate_dual_gate,
    grouped_cross_validate,
)
from ssv4d.runtime import install_runtime

__all__ = [
    "CompositeRiskGate",
    "DualThresholdGate",
    "PerturbationPlan",
    "SGLangShadowClient",
    "calibrate_and_evaluate",
    "calibrate_dual_gate",
    "grouped_cross_validate",
    "install_runtime",
    "make_perturbation_plans",
]
