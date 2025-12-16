"""Compilation utilities for GFDSL."""

from .canonicalize import canonicalize, canonical_node_dict, full_hash, structure_hash
from .lower import (
    ColumnEvaluator,
    DenseEvaluator,
    LinearContribution,
    OperatorEvaluator,
    lower_program,
)
from .legacy_adapter import LegacyBasisElement, linear_contribution_to_legacy_basis
from .types import CoeffSlot, CompileContext
from .validate import GFDSLValidationError, validate_program

__all__ = [
    "canonicalize",
    "canonical_node_dict",
    "full_hash",
    "structure_hash",
    "ColumnEvaluator",
    "DenseEvaluator",
    "LinearContribution",
    "OperatorEvaluator",
    "lower_program",
    "CoeffSlot",
    "CompileContext",
    "LegacyBasisElement",
    "linear_contribution_to_legacy_basis",
    "GFDSLValidationError",
    "validate_program",
]
