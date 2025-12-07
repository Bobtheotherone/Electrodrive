from __future__ import annotations

"""
Thin compatibility shim re-exporting BasisOperator from electrodrive.images.basis.

The operator implementation now lives alongside the basis definitions to keep
batched kernels and grouping logic in one place.
"""

from electrodrive.images.basis import BasisOperator

__all__ = ["BasisOperator"]
