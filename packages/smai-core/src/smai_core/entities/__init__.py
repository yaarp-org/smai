"""Methodology entity public surface.

Re-exports every entity from ``designs/smai/01-data-model.md`` §2's inventory.
Consumers should import from ``smai_core`` (or this subpackage) rather than
from individual modules — the module split is an implementation detail.
"""

from smai_core.entities.compute_requirements import ComputeRequirements
from smai_core.entities.conditions import ControlledConditions
from smai_core.entities.experiment import (
    ExperimentDefinition,
    FactorModel,
    VerifiedExperimentDefinition,
)
from smai_core.entities.factor import Entry, Factor, Level
from smai_core.entities.metric import (
    AtomicMetricEntry,
    AtomicMetricRef,
    MetricRef,
    MetricRefAdapter,
    MetricRefNotRegistered,
    MetricRegistry,
    MetricRegistryEntry,
    ParametricFamily,
    ParametricMetricRef,
)
from smai_core.entities.numeric import NumericValue
from smai_core.entities.registries import Registries
from smai_core.entities.technique import (
    FidelityAnchor,
    FidelityAnchorAdapter,
    PaperFidelityAnchor,
    ProposalFidelityAnchor,
    ReviewerAttestedFidelityAnchor,
    TechniqueParams,
    TechniqueParamValue,
    TechniqueRef,
)
from smai_core.entities.validation import (
    AggregationRule,
    ComparisonRule,
    TrendCheck,
    ValidationCriteria,
)
from smai_core.entities.validation_report import (
    RuleSeverity,
    ValidationError,
    ValidationReport,
    VerificationFinding,
)

__all__ = [
    "AggregationRule",
    "AtomicMetricEntry",
    "AtomicMetricRef",
    "ComparisonRule",
    "ComputeRequirements",
    "ControlledConditions",
    "Entry",
    "ExperimentDefinition",
    "Factor",
    "FactorModel",
    "FidelityAnchor",
    "FidelityAnchorAdapter",
    "Level",
    "MetricRef",
    "MetricRefAdapter",
    "MetricRefNotRegistered",
    "MetricRegistry",
    "MetricRegistryEntry",
    "NumericValue",
    "PaperFidelityAnchor",
    "ParametricFamily",
    "ParametricMetricRef",
    "ProposalFidelityAnchor",
    "Registries",
    "ReviewerAttestedFidelityAnchor",
    "RuleSeverity",
    "TechniqueParamValue",
    "TechniqueParams",
    "TechniqueRef",
    "TrendCheck",
    "ValidationCriteria",
    "ValidationError",
    "ValidationReport",
    "VerificationFinding",
    "VerifiedExperimentDefinition",
]
