"""
GFDSL node definitions.

These are structural stubs for the GFDSL AST. Evaluation/lowering will be added
in later milestones; for now the focus is on serialization, validation, and
canonicalization stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterable, Optional, Tuple

import torch

from .constraints import GroupInfo
from .registry import register_node
from .types import Param
from electrodrive.gfdsl.eval.kernels_complex import complex_conjugate_pair_columns
from electrodrive.gfdsl.eval.kernels_real import coulomb_potential_real, dipole_basis_real

if TYPE_CHECKING:
    from electrodrive.gfdsl.compile.lower import (
        DenseEvaluator,
        LinearContribution,
        OperatorEvaluator,
    )
    from electrodrive.gfdsl.compile.types import CoeffSlot, CompileContext


def _deserialize_children(raw_children: Iterable[Any]) -> Tuple["GFNode", ...]:
    from .registry import make_node

    deserialized = []
    for child in raw_children or []:
        if isinstance(child, GFNode):
            deserialized.append(child)
        elif isinstance(child, dict):
            node_type = child.get("node_type")
            payload = {k: v for k, v in child.items() if k != "node_type"}
            deserialized.append(make_node(node_type, **payload))
        else:
            raise TypeError(f"Unsupported child type for GFNode: {type(child)}")
    return tuple(deserialized)


def _deserialize_params(raw_params: Dict[str, Any]) -> Dict[str, Param]:
    params: Dict[str, Param] = {}
    for key, value in (raw_params or {}).items():
        if isinstance(value, Param):
            params[key] = value
        elif isinstance(value, dict):
            params[key] = Param.from_json_dict(value)
        else:
            raise TypeError(f"Unsupported param type for key '{key}': {type(value)}")
    return params


@dataclass
class GFNode:
    """Base AST node."""

    children: Tuple["GFNode", ...] = field(default_factory=tuple)
    params: Dict[str, Param] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    group_info: Optional[GroupInfo] = None

    node_type: ClassVar[str] = "base"

    def __post_init__(self) -> None:
        self.children = tuple(self.children)
        self.params = dict(self.params)
        if isinstance(self.group_info, dict):
            self.group_info = GroupInfo.from_json_dict(self.group_info)

    def validate(self, ctx: Any) -> None:
        """Lightweight structural validation hook."""
        return None

    def lower(self, ctx: CompileContext) -> LinearContribution:
        raise NotImplementedError(f"Lowering not implemented for node type '{self.node_type}'")

    def to_json_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "node_type": self.node_type,
            "children": [child.to_json_dict() for child in self.children],
            "params": {k: v.to_json_dict() for k, v in self.params.items()},
            "meta": self.meta or {},
        }
        if self.group_info is not None:
            data["group_info"] = self.group_info.to_json_dict()
        return data

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "GFNode":
        children = _deserialize_children(data.get("children", []))
        params = _deserialize_params(data.get("params", {}))
        meta = data.get("meta", {}) or {}
        group_info_raw = data.get("group_info")
        group_info = (
            GroupInfo.from_json_dict(group_info_raw)
            if group_info_raw is not None
            else None
        )
        return cls(children=children, params=params, meta=meta, group_info=group_info)

    def canonical_dict(
        self, include_raw: bool = False, quantization: float = 1e-12
    ) -> Dict[str, Any]:
        from electrodrive.gfdsl.compile.canonicalize import canonical_node_dict

        return canonical_node_dict(self, include_raw=include_raw, quantization=quantization)

    def structure_hash(self) -> str:
        from electrodrive.gfdsl.compile.canonicalize import structure_hash

        return structure_hash(self)

    def full_hash(self) -> str:
        from electrodrive.gfdsl.compile.canonicalize import full_hash

        return full_hash(self)

    def node_hash(self) -> str:
        return self.structure_hash()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.node_type}, children={len(self.children)})"


def _axis_index(axis: str) -> int:
    if axis not in ("x", "y", "z"):
        raise ValueError(f"Unsupported axis '{axis}', expected one of x, y, z")
    return {"x": 0, "y": 1, "z": 2}[axis]


def _clone_slots_with_group_policy(
    slots: Iterable["CoeffSlot"],
    family_name: str,
    group_policy: str,
    motif_index: int,
) -> list["CoeffSlot"]:
    from electrodrive.gfdsl.compile.types import CoeffSlot

    cloned: list[CoeffSlot] = []
    for slot in slots:
        base_group = slot.group_info
        if group_policy == "override":
            conductor_id = base_group.conductor_id if base_group is not None else -1
            extra = dict(getattr(base_group, "extra", {}) or {})
            group = GroupInfo(
                conductor_id=conductor_id,
                family_name=family_name,
                motif_index=motif_index,
                extra=extra,
            )
        else:
            if base_group is None:
                group = GroupInfo(family_name=family_name, motif_index=motif_index)
            else:
                group = base_group
        cloned.append(
            CoeffSlot(
                slot_id=slot.slot_id,
                dim=slot.dim,
                regularizer=dict(slot.regularizer),
                group_info=group,
                meta=dict(slot.meta),
            )
        )
    return cloned


@register_node
@dataclass
class RealImageChargeNode(GFNode):
    node_type: ClassVar[str] = "real_image_charge"

    def validate(self, ctx: Any) -> None:
        if "charge" in self.params:
            raise ValueError(
                "RealImageChargeNode no longer accepts 'charge'. "
                "Use solver coefficient slots (unknown) or FixedChargeNode (fixed)."
            )
        required = ("position",)
        for key in required:
            if key not in self.params:
                raise ValueError(f"RealImageChargeNode missing required param '{key}'")

    def lower(self, ctx: CompileContext) -> LinearContribution:
        from electrodrive.gfdsl.compile.lower import (
            DenseEvaluator,
            LinearContribution,
            OperatorEvaluator,
        )
        from electrodrive.gfdsl.compile.types import CoeffSlot, CompileContext

        device = ctx.device
        dtype = ctx.dtype
        pos = self.params["position"].value(device=device, dtype=dtype).reshape(1, 3)

        base_hash = self.full_hash()
        group = self.group_info or GroupInfo(family_name="real_charge")
        slot = CoeffSlot(
            slot_id=f"{base_hash}:0",
            dim=1,
            group_info=group,
            meta={"node_type": self.node_type},
        )

        def dense_fn(X: torch.Tensor) -> torch.Tensor:
            return coulomb_potential_real(X, pos)

        if ctx.eval_backend == "operator":

            def matvec_fn(w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
                w_scalar = w.to(device=X.device, dtype=X.dtype).reshape(-1)
                phi = coulomb_potential_real(X, pos).squeeze(-1)
                return phi * w_scalar[0]

            def rmatvec_fn(r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
                phi = coulomb_potential_real(X, pos).squeeze(-1)
                r_cast = r.to(device=X.device, dtype=X.dtype).reshape(-1)
                grad = torch.dot(r_cast, phi)
                return grad.view(1)

            evaluator = OperatorEvaluator(1, matvec_fn, rmatvec_fn, dense_fn=dense_fn)
        else:
            evaluator = DenseEvaluator(1, dense_fn)

        return LinearContribution(slots=[slot], evaluator=evaluator)


@register_node
@dataclass
class FixedChargeNode(GFNode):
    node_type: ClassVar[str] = "fixed_charge"

    def validate(self, ctx: Any) -> None:
        for key in ("position", "charge"):
            if key not in self.params:
                raise ValueError(f"FixedChargeNode missing required param '{key}'")

    def lower(self, ctx: CompileContext) -> LinearContribution:
        from electrodrive.gfdsl.compile.lower import LinearContribution, OperatorEvaluator

        device = ctx.device
        dtype = ctx.dtype
        pos = self.params["position"].value(device=device, dtype=dtype).reshape(1, 3)
        charge = self.params["charge"].value(device=device, dtype=dtype).reshape(1)

        def dense_fn(X: torch.Tensor) -> torch.Tensor:
            return torch.zeros(X.shape[0], 0, device=X.device, dtype=X.dtype)

        def fixed_fn(X: torch.Tensor) -> torch.Tensor:
            phi = coulomb_potential_real(X, pos).squeeze(-1)
            return phi * charge.view(-1)[0]

        def matvec_fn(w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
            return torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)

        def rmatvec_fn(r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
            return torch.zeros(0, device=X.device, dtype=X.dtype)

        evaluator = OperatorEvaluator(0, matvec_fn, rmatvec_fn, dense_fn=dense_fn)
        return LinearContribution(slots=[], evaluator=evaluator, fixed_term=fixed_fn)


@register_node
@dataclass
class FixedDipoleNode(GFNode):
    node_type: ClassVar[str] = "fixed_dipole"

    def validate(self, ctx: Any) -> None:
        for key in ("position", "moment"):
            if key not in self.params:
                raise ValueError(f"FixedDipoleNode missing required param '{key}'")

    def lower(self, ctx: CompileContext) -> LinearContribution:
        from electrodrive.gfdsl.compile.lower import LinearContribution, OperatorEvaluator

        device = ctx.device
        dtype = ctx.dtype
        pos = self.params["position"].value(device=device, dtype=dtype).reshape(1, 3)
        moment = self.params["moment"].value(device=device, dtype=dtype).reshape(1, 3)

        def dense_fn(X: torch.Tensor) -> torch.Tensor:
            return torch.zeros(X.shape[0], 0, device=X.device, dtype=X.dtype)

        def fixed_fn(X: torch.Tensor) -> torch.Tensor:
            basis = dipole_basis_real(X, pos)
            return torch.sum(basis * moment.view(1, 3), dim=1)

        def matvec_fn(w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
            return torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)

        def rmatvec_fn(r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
            return torch.zeros(0, device=X.device, dtype=X.dtype)

        evaluator = OperatorEvaluator(0, matvec_fn, rmatvec_fn, dense_fn=dense_fn)
        return LinearContribution(slots=[], evaluator=evaluator, fixed_term=fixed_fn)


@register_node
@dataclass
class ComplexImageChargeNode(GFNode):
    node_type: ClassVar[str] = "complex_image_charge"

    def validate(self, ctx: Any) -> None:
        required = ("x", "y", "a", "b")
        for key in required:
            if key not in self.params:
                raise ValueError(f"ComplexImageChargeNode missing required param '{key}'")
        b_param = self.params["b"]
        if b_param.transform.__class__.__name__.lower().startswith("identity"):
            raise ValueError("ComplexImageChargeNode param 'b' must use a positive transform")

    def lower(self, ctx: CompileContext) -> LinearContribution:
        raise ValueError("ComplexImageChargeNode must be lowered via ConjugatePairNode")


@register_node
@dataclass
class DipoleNode(GFNode):
    node_type: ClassVar[str] = "dipole"

    def validate(self, ctx: Any) -> None:
        if "moment" in self.params:
            raise ValueError(
                "DipoleNode no longer accepts 'moment'. "
                "Use 3 coefficient slots (unknown) or FixedDipoleNode (fixed)."
            )
        for key in ("position",):
            if key not in self.params:
                raise ValueError(f"DipoleNode missing required param '{key}'")

    def lower(self, ctx: CompileContext) -> LinearContribution:
        from electrodrive.gfdsl.compile.lower import (
            DenseEvaluator,
            LinearContribution,
            OperatorEvaluator,
        )
        from electrodrive.gfdsl.compile.types import CoeffSlot, CompileContext

        device = ctx.device
        dtype = ctx.dtype
        pos = self.params["position"].value(device=device, dtype=dtype).reshape(-1, 3)
        if pos.shape[0] != 1:
            raise ValueError("DipoleNode lowering currently supports exactly one position")

        expected_cols = 3

        base_hash = self.full_hash()
        group = self.group_info or GroupInfo(family_name="dipole")
        slots = [
            CoeffSlot(
                slot_id=f"{base_hash}:{idx}",
                dim=1,
                group_info=group,
                meta={"node_type": self.node_type, "component": comp},
            )
            for idx, comp in enumerate(("px", "py", "pz"))
        ]

        def dense_fn(X: torch.Tensor) -> torch.Tensor:
            return dipole_basis_real(X, pos)

        if ctx.eval_backend == "operator":

            def matvec_fn(w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
                basis = dipole_basis_real(X, pos)
                w_vec = w.to(device=X.device, dtype=X.dtype).reshape(-1)
                return torch.sum(basis * w_vec.view(1, 3), dim=1)

            def rmatvec_fn(r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
                basis = dipole_basis_real(X, pos)
                r_cast = r.to(device=X.device, dtype=X.dtype).reshape(-1)
                return basis.transpose(0, 1) @ r_cast

            evaluator = OperatorEvaluator(expected_cols, matvec_fn, rmatvec_fn, dense_fn=dense_fn)
        else:
            evaluator = DenseEvaluator(expected_cols, dense_fn)

        return LinearContribution(slots=slots, evaluator=evaluator)


@register_node
@dataclass
class ComplexDipoleNode(GFNode):
    node_type: ClassVar[str] = "complex_dipole"

    def validate(self, ctx: Any) -> None:
        for key in ("position", "moment"):
            if key not in self.params:
                raise ValueError(f"ComplexDipoleNode missing required param '{key}'")


@register_node
@dataclass
class MultipoleNode(GFNode):
    node_type: ClassVar[str] = "multipole"

    def validate(self, ctx: Any) -> None:
        for key in ("L", "m", "coeff", "position"):
            if key not in self.params:
                raise ValueError(f"MultipoleNode missing required param '{key}'")
        L_val = int(self.params["L"].value().item())
        m_val = int(self.params["m"].value().item())
        if L_val < 0:
            raise ValueError("MultipoleNode requires L >= 0")
        if m_val < -L_val or m_val > L_val:
            raise ValueError("MultipoleNode requires m within [-L, L]")


@register_node
@dataclass
class InterfacePoleNode(GFNode):
    node_type: ClassVar[str] = "interface_pole"

    def validate(self, ctx: Any) -> None:
        for key in ("mode_id", "k_pole", "residue"):
            if key not in self.params:
                raise ValueError(f"InterfacePoleNode missing required param '{key}'")

    def lower(self, ctx: CompileContext) -> LinearContribution:
        from electrodrive.gfdsl.compile.lower import (
            DenseEvaluator,
            LinearContribution,
            OperatorEvaluator,
        )
        from electrodrive.gfdsl.compile.types import CoeffSlot, CompileContext
        from electrodrive.gfdsl.eval.layered import interface_pole_columns, resolve_layered_frame

        device = ctx.device
        dtype = ctx.dtype
        k_pole = self.params["k_pole"].value(device=device, dtype=dtype).reshape(-1)
        residue = self.params["residue"].value(device=device, dtype=dtype).reshape(-1)
        if k_pole.numel() == 0 or residue.numel() == 0:
            raise ValueError("InterfacePoleNode requires non-empty k_pole and residue params")
        if residue.numel() == 1 and k_pole.numel() > 1:
            residue = residue.expand_as(k_pole)
        if k_pole.numel() == 1 and residue.numel() > 1:
            k_pole = k_pole.expand_as(residue)
        if k_pole.numel() != residue.numel():
            raise ValueError("InterfacePoleNode k_pole and residue must have matching lengths")

        direction = float(self.meta.get("direction", 1.0))
        depths = k_pole * direction
        src_xy, z_interface = resolve_layered_frame(ctx, self.meta, device=device, dtype=dtype)

        base_hash = self.full_hash()
        group = self.group_info or GroupInfo(family_name="interface_pole")
        mode_id = None
        try:
            mode_id = int(self.params["mode_id"].value().view(-1)[0].item())
        except Exception:
            mode_id = None
        slots = [
            CoeffSlot(
                slot_id=f"{base_hash}:{idx}",
                dim=1,
                group_info=group,
                meta={"node_type": self.node_type, "mode_id": mode_id},
            )
            for idx in range(int(k_pole.numel()))
        ]

        def dense_fn(X: torch.Tensor) -> torch.Tensor:
            return interface_pole_columns(
                X, src_xy=src_xy, z_interface=z_interface, depths=depths, residues=residue
            )

        if ctx.eval_backend == "operator":

            def matvec_fn(w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
                Phi = dense_fn(X)
                w_cast = w.to(device=X.device, dtype=X.dtype).reshape(-1)
                return Phi @ w_cast

            def rmatvec_fn(r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
                Phi = dense_fn(X)
                r_cast = r.to(device=X.device, dtype=X.dtype).reshape(-1)
                return Phi.transpose(0, 1) @ r_cast

            evaluator = OperatorEvaluator(len(slots), matvec_fn, rmatvec_fn, dense_fn=dense_fn)
        else:
            evaluator = DenseEvaluator(len(slots), dense_fn)

        return LinearContribution(slots=slots, evaluator=evaluator)


@register_node
@dataclass
class BranchCutApproxNode(GFNode):
    node_type: ClassVar[str] = "branch_cut_approx"

    def validate(self, ctx: Any) -> None:
        kind = (self.meta or {}).get("kind")
        if kind is None:
            raise ValueError("BranchCutApproxNode requires meta['kind']")
        allowed = {"quadrature_hankel", "exp_sum"}
        if kind not in allowed:
            raise ValueError(f"BranchCutApproxNode kind '{kind}' not in {sorted(allowed)}")
        for key in ("depths", "weights"):
            if key not in self.params:
                raise ValueError(f"BranchCutApproxNode requires param '{key}' for kind '{kind}'")

    def lower(self, ctx: CompileContext) -> LinearContribution:
        from electrodrive.gfdsl.compile.lower import (
            DenseEvaluator,
            LinearContribution,
            OperatorEvaluator,
        )
        from electrodrive.gfdsl.compile.types import CoeffSlot, CompileContext
        from electrodrive.gfdsl.eval.layered import branch_cut_exp_sum_columns, resolve_layered_frame

        device = ctx.device
        dtype = ctx.dtype
        depths = self.params["depths"].value(device=device, dtype=dtype).reshape(-1)
        weights = self.params["weights"].value(device=device, dtype=dtype).reshape(-1)
        if depths.numel() == 0 or weights.numel() == 0:
            raise ValueError("BranchCutApproxNode requires non-empty depths and weights")
        if weights.numel() == 1 and depths.numel() > 1:
            weights = weights.expand_as(depths)
        if depths.numel() == 1 and weights.numel() > 1:
            depths = depths.expand_as(weights)
        if depths.numel() != weights.numel():
            raise ValueError("BranchCutApproxNode depths and weights must have matching lengths")

        src_xy, z_interface = resolve_layered_frame(ctx, self.meta, device=device, dtype=dtype)

        base_hash = self.full_hash()
        group = self.group_info or GroupInfo(family_name="branch_cut")
        slots = [
            CoeffSlot(
                slot_id=f"{base_hash}:{idx}",
                dim=1,
                group_info=group,
                meta={"node_type": self.node_type, "kind": (self.meta or {}).get("kind")},
            )
            for idx in range(int(depths.numel()))
        ]

        def dense_fn(X: torch.Tensor) -> torch.Tensor:
            return branch_cut_exp_sum_columns(
                X, src_xy=src_xy, z_interface=z_interface, depths=depths, weights=weights
            )

        if ctx.eval_backend == "operator":

            def matvec_fn(w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
                Phi = dense_fn(X)
                w_cast = w.to(device=X.device, dtype=X.dtype).reshape(-1)
                return Phi @ w_cast

            def rmatvec_fn(r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
                Phi = dense_fn(X)
                r_cast = r.to(device=X.device, dtype=X.dtype).reshape(-1)
                return Phi.transpose(0, 1) @ r_cast

            evaluator = OperatorEvaluator(len(slots), matvec_fn, rmatvec_fn, dense_fn=dense_fn)
        else:
            evaluator = DenseEvaluator(len(slots), dense_fn)

        return LinearContribution(slots=slots, evaluator=evaluator)


@register_node
@dataclass
class MirrorAcrossPlaneNode(GFNode):
    node_type: ClassVar[str] = "mirror_across_plane"

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.children) > 1:
            raise ValueError("MirrorAcrossPlaneNode expects at most one child")

    def validate(self, ctx: Any) -> None:
        if "z0" not in self.params:
            raise ValueError("MirrorAcrossPlaneNode requires param 'z0'")
        if len(self.children) != 1:
            raise ValueError("MirrorAcrossPlaneNode must wrap exactly one child")

    def lower(self, ctx: CompileContext) -> LinearContribution:
        from electrodrive.gfdsl.compile.lower import (
            DenseEvaluator,
            LinearContribution,
            OperatorEvaluator,
        )

        if len(self.children) != 1:
            raise ValueError("MirrorAcrossPlaneNode must wrap exactly one child")

        child = self.children[0]
        if not isinstance(
            child, (RealImageChargeNode, DipoleNode, ConjugatePairNode, FixedChargeNode, FixedDipoleNode, SumNode)
        ):
            raise NotImplementedError(
                "MirrorAcrossPlaneNode currently supports RealImageChargeNode, "
                "DipoleNode, ConjugatePairNode, FixedChargeNode, FixedDipoleNode, or SumNode containing those children"
            )
        if isinstance(child, SumNode):
            unsupported = [
                c
                for c in child.children
                if not isinstance(
                    c, (RealImageChargeNode, DipoleNode, ConjugatePairNode, FixedChargeNode, FixedDipoleNode)
                )
            ]
            if unsupported:
                raise NotImplementedError(
                    "MirrorAcrossPlaneNode Sum children must be real_image_charge, dipole, conjugate_pair, fixed_charge, or fixed_dipole"
                )

        device = ctx.device
        dtype = ctx.dtype
        z0 = self.params["z0"].value(device=device, dtype=dtype).reshape(-1)
        if z0.numel() != 1:
            raise ValueError("MirrorAcrossPlaneNode param 'z0' must be scalar")
        z0_scalar = z0.view(1)[0]

        sign = float(self.meta.get("sign", -1.0))
        mode = self.meta.get("mode", "composite_single_slot")
        if mode != "composite_single_slot":
            raise NotImplementedError(
                f"MirrorAcrossPlaneNode mode '{mode}' not implemented (only composite_single_slot supported)"
            )
        group_policy = self.meta.get("group_policy", "inherit")
        motif_index = int(self.meta.get("motif_index", 0))

        child_contrib = child.lower(ctx)
        slots = _clone_slots_with_group_policy(
            child_contrib.slots, "mirror_plane", group_policy, motif_index
        )
        child_evaluator = child_contrib.evaluator
        child_has_fixed = child_contrib.fixed_term is not None

        def _node_K(node: GFNode) -> int:
            if isinstance(node, RealImageChargeNode):
                return 1
            if isinstance(node, DipoleNode):
                return 3
            if isinstance(node, ConjugatePairNode):
                return 2
            if isinstance(node, (FixedChargeNode, FixedDipoleNode)):
                return 0
            if isinstance(node, SumNode):
                return sum(_node_K(c) for c in node.children)
            return 0

        def _has_fixed(node: GFNode) -> bool:
            if isinstance(node, (FixedChargeNode, FixedDipoleNode)):
                return True
            if isinstance(node, SumNode):
                return any(_has_fixed(c) for c in node.children)
            return False

        def _reflected_columns_for_node(node: GFNode, X: torch.Tensor) -> torch.Tensor:
            if isinstance(node, RealImageChargeNode):
                pos = node.params["position"].value(device=device, dtype=dtype).reshape(1, 3)
                reflected_pos = pos.clone()
                reflected_pos[..., 2] = 2.0 * z0_scalar - pos[..., 2]
                phi_ref = coulomb_potential_real(X, reflected_pos)
                return torch.as_tensor(sign, device=X.device, dtype=X.dtype) * phi_ref
            if isinstance(node, DipoleNode):
                pos = node.params["position"].value(device=device, dtype=dtype).reshape(1, 3)
                reflected_pos = pos.clone()
                reflected_pos[..., 2] = 2.0 * z0_scalar - pos[..., 2]
                basis = dipole_basis_real(X, reflected_pos).reshape(X.shape[0], 3)
                scaled = torch.stack(
                    (
                        torch.as_tensor(sign, device=X.device, dtype=X.dtype) * basis[:, 0],
                        torch.as_tensor(sign, device=X.device, dtype=X.dtype) * basis[:, 1],
                        torch.as_tensor(-sign, device=X.device, dtype=X.dtype) * basis[:, 2],
                    ),
                    dim=1,
                )
                return scaled
            if isinstance(node, ConjugatePairNode):
                complex_child = node.children[0]
                assert isinstance(complex_child, ComplexImageChargeNode)
                x = complex_child.params["x"].value(device=device, dtype=dtype).reshape(-1)
                y = complex_child.params["y"].value(device=device, dtype=dtype).reshape(-1)
                a = complex_child.params["a"].value(device=device, dtype=dtype).reshape(-1)
                b = complex_child.params["b"].value(device=device, dtype=dtype).reshape(-1)
                reflected_a = 2.0 * z0_scalar - a
                xyab = torch.stack((x, y, reflected_a, b), dim=-1).reshape(1, 4)
                phi_ref = complex_conjugate_pair_columns(X, xyab)
                return torch.as_tensor(sign, device=X.device, dtype=X.dtype) * phi_ref
            if isinstance(node, (FixedChargeNode, FixedDipoleNode)):
                return torch.zeros(X.shape[0], 0, device=X.device, dtype=X.dtype)
            if isinstance(node, SumNode):
                cols = [_reflected_columns_for_node(sub, X) for sub in node.children]
                return torch.cat(cols, dim=1) if len(cols) > 1 else (cols[0] if cols else torch.zeros(X.shape[0], 0, device=X.device, dtype=X.dtype))
            raise RuntimeError("Unsupported child type for reflection")

        def dense_fn(X: torch.Tensor) -> torch.Tensor:
            base_cols = child_evaluator.eval_columns(X)
            reflected_cols = _reflected_columns_for_node(child, X)
            return base_cols + reflected_cols

        def _reflection_matvec(node: GFNode, w_vec: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
            if isinstance(node, RealImageChargeNode):
                pos = node.params["position"].value(device=device, dtype=dtype).reshape(1, 3)
                reflected_pos = pos.clone()
                reflected_pos[..., 2] = 2.0 * z0_scalar - pos[..., 2]
                phi_ref = coulomb_potential_real(X, reflected_pos)
                phi_ref = phi_ref.squeeze(-1)
                return torch.as_tensor(sign, device=X.device, dtype=X.dtype) * phi_ref * w_vec[0]
            if isinstance(node, DipoleNode):
                pos = node.params["position"].value(device=device, dtype=dtype).reshape(1, 3)
                reflected_pos = pos.clone()
                reflected_pos[..., 2] = 2.0 * z0_scalar - pos[..., 2]
                basis = dipole_basis_real(X, reflected_pos).reshape(X.shape[0], 3)
                w_scaled = torch.stack(
                    (
                        torch.as_tensor(sign, device=X.device, dtype=X.dtype) * w_vec[0],
                        torch.as_tensor(sign, device=X.device, dtype=X.dtype) * w_vec[1],
                        torch.as_tensor(-sign, device=X.device, dtype=X.dtype) * w_vec[2],
                    )
                )
                return torch.sum(basis * w_scaled.view(1, 3), dim=1)
            if isinstance(node, ConjugatePairNode):
                complex_child = node.children[0]
                assert isinstance(complex_child, ComplexImageChargeNode)
                x = complex_child.params["x"].value(device=device, dtype=dtype).reshape(-1)
                y = complex_child.params["y"].value(device=device, dtype=dtype).reshape(-1)
                a = complex_child.params["a"].value(device=device, dtype=dtype).reshape(-1)
                b = complex_child.params["b"].value(device=device, dtype=dtype).reshape(-1)
                reflected_a = 2.0 * z0_scalar - a
                xyab = torch.stack((x, y, reflected_a, b), dim=-1).reshape(1, 4)
                cols = complex_conjugate_pair_columns(X, xyab)
                return torch.as_tensor(sign, device=X.device, dtype=X.dtype) * (cols @ w_vec.view(-1))
            if isinstance(node, (FixedChargeNode, FixedDipoleNode)):
                return torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
            if isinstance(node, SumNode):
                out = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
                offset = 0
                for sub in node.children:
                    k = _node_K(sub)
                    w_slice = w_vec[offset : offset + k]
                    out = out + _reflection_matvec(sub, w_slice, X)
                    offset += k
                return out
            raise RuntimeError("Unsupported child type for reflection")

        def _reflection_rmatvec(node: GFNode, r_vec: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
            if isinstance(node, RealImageChargeNode):
                pos = node.params["position"].value(device=device, dtype=dtype).reshape(1, 3)
                reflected_pos = pos.clone()
                reflected_pos[..., 2] = 2.0 * z0_scalar - pos[..., 2]
                phi_ref = coulomb_potential_real(X, reflected_pos).squeeze(-1)
                grad = torch.dot(r_vec, phi_ref)
                return grad.view(1) * torch.as_tensor(sign, device=X.device, dtype=X.dtype)
            if isinstance(node, DipoleNode):
                pos = node.params["position"].value(device=device, dtype=dtype).reshape(1, 3)
                reflected_pos = pos.clone()
                reflected_pos[..., 2] = 2.0 * z0_scalar - pos[..., 2]
                basis = dipole_basis_real(X, reflected_pos).reshape(X.shape[0], 3)
                grads = basis.transpose(0, 1) @ r_vec
                scale = torch.as_tensor([sign, sign, -sign], device=X.device, dtype=X.dtype)
                return grads * scale
            if isinstance(node, ConjugatePairNode):
                complex_child = node.children[0]
                assert isinstance(complex_child, ComplexImageChargeNode)
                x = complex_child.params["x"].value(device=device, dtype=dtype).reshape(-1)
                y = complex_child.params["y"].value(device=device, dtype=dtype).reshape(-1)
                a = complex_child.params["a"].value(device=device, dtype=dtype).reshape(-1)
                b = complex_child.params["b"].value(device=device, dtype=dtype).reshape(-1)
                reflected_a = 2.0 * z0_scalar - a
                xyab = torch.stack((x, y, reflected_a, b), dim=-1).reshape(1, 4)
                cols = complex_conjugate_pair_columns(X, xyab)
                grads = cols.transpose(0, 1) @ r_vec
                return torch.as_tensor(sign, device=X.device, dtype=X.dtype) * grads
            if isinstance(node, (FixedChargeNode, FixedDipoleNode)):
                return torch.zeros(0, device=X.device, dtype=X.dtype)
            if isinstance(node, SumNode):
                grads = []
                offset = 0
                for sub in node.children:
                    k = _node_K(sub)
                    grads.append(_reflection_rmatvec(sub, r_vec, X)[:k])
                    offset += k
                return torch.cat(grads, dim=0) if grads else torch.zeros(0, device=X.device, dtype=X.dtype)
            raise RuntimeError("Unsupported child type for reflection")

        def _fixed_term_for_node(node: GFNode, X: torch.Tensor) -> torch.Tensor:
            if isinstance(node, FixedChargeNode):
                pos = node.params["position"].value(device=device, dtype=dtype).reshape(1, 3)
                charge = node.params["charge"].value(device=device, dtype=dtype).reshape(1)
                phi = coulomb_potential_real(X, pos).squeeze(-1)
                return phi * charge.view(-1)[0]
            if isinstance(node, FixedDipoleNode):
                pos = node.params["position"].value(device=device, dtype=dtype).reshape(1, 3)
                moment = node.params["moment"].value(device=device, dtype=dtype).reshape(1, 3)
                basis = dipole_basis_real(X, pos)
                return torch.sum(basis * moment.view(1, 3), dim=1)
            if isinstance(node, SumNode):
                parts = [_fixed_term_for_node(sub, X) for sub in node.children]
                return sum(parts) if parts else torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
            return torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)

        def _fixed_term_reflected(node: GFNode, X: torch.Tensor) -> torch.Tensor:
            if isinstance(node, FixedChargeNode):
                pos = node.params["position"].value(device=device, dtype=dtype).reshape(1, 3)
                reflected_pos = pos.clone()
                reflected_pos[..., 2] = 2.0 * z0_scalar - pos[..., 2]
                charge = node.params["charge"].value(device=device, dtype=dtype).reshape(1)
                phi = coulomb_potential_real(X, reflected_pos).squeeze(-1)
                return torch.as_tensor(sign, device=X.device, dtype=X.dtype) * phi * charge.view(-1)[0]
            if isinstance(node, FixedDipoleNode):
                pos = node.params["position"].value(device=device, dtype=dtype).reshape(1, 3)
                moment = node.params["moment"].value(device=device, dtype=dtype).reshape(1, 3)
                reflected_pos = pos.clone()
                reflected_pos[..., 2] = 2.0 * z0_scalar - pos[..., 2]
                reflected_moment = moment.clone()
                reflected_moment[..., 2] = -reflected_moment[..., 2]
                basis = dipole_basis_real(X, reflected_pos)
                pot = torch.sum(basis * reflected_moment.view(1, 3), dim=1)
                return torch.as_tensor(sign, device=X.device, dtype=X.dtype) * pot
            if isinstance(node, SumNode):
                parts = [_fixed_term_reflected(sub, X) for sub in node.children]
                return sum(parts) if parts else torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
            return torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)

        if ctx.eval_backend == "operator":

            def matvec_fn(w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
                w_cast = w.to(device=X.device, dtype=X.dtype).reshape(-1)
                base = child_evaluator.matvec(w_cast, X)
                reflected = _reflection_matvec(child, w_cast, X)
                return base + reflected

            def rmatvec_fn(r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
                r_cast = r.to(device=X.device, dtype=X.dtype).reshape(-1)
                base = child_evaluator.rmatvec(r_cast, X)
                reflected = _reflection_rmatvec(child, r_cast, X)
                return base + reflected

            evaluator = OperatorEvaluator(child_contrib.evaluator.K, matvec_fn, rmatvec_fn, dense_fn=dense_fn)
            evaluator._child_evaluator = child_evaluator  # debug hook for tests
        else:
            evaluator = DenseEvaluator(child_contrib.evaluator.K, dense_fn)

        fixed_term_fn = None
        if _has_fixed(child) or child_has_fixed:
            def fixed_term_fn(X: torch.Tensor) -> torch.Tensor:
                return _fixed_term_for_node(child, X) + _fixed_term_reflected(child, X)

        return LinearContribution(slots=slots, evaluator=evaluator, fixed_term=fixed_term_fn)


@register_node
@dataclass
class ImageLadderNode(GFNode):
    node_type: ClassVar[str] = "image_ladder"

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.children) > 1:
            raise ValueError("ImageLadderNode expects at most one child")

    def validate(self, ctx: Any) -> None:
        required = ("step", "count")
        for key in required:
            if key not in self.params:
                raise ValueError(f"ImageLadderNode missing required param '{key}'")
        if len(self.children) != 1:
            raise ValueError("ImageLadderNode must wrap exactly one child")

    def lower(self, ctx: CompileContext) -> LinearContribution:
        from electrodrive.gfdsl.compile.lower import (
            DenseEvaluator,
            LinearContribution,
            OperatorEvaluator,
        )

        if len(self.children) != 1:
            raise ValueError("ImageLadderNode must wrap exactly one child")

        child = self.children[0]
        if not isinstance(
            child, (RealImageChargeNode, DipoleNode, ConjugatePairNode, FixedChargeNode, FixedDipoleNode, SumNode)
        ):
            raise NotImplementedError(
                "ImageLadderNode currently supports RealImageChargeNode, DipoleNode, ConjugatePairNode, "
                "FixedChargeNode, FixedDipoleNode, or SumNode containing those children"
            )
        if isinstance(child, SumNode):
            unsupported = [
                c
                for c in child.children
                if not isinstance(
                    c, (RealImageChargeNode, DipoleNode, ConjugatePairNode, FixedChargeNode, FixedDipoleNode)
                )
            ]
            if unsupported:
                raise NotImplementedError(
                    "ImageLadderNode Sum children must be real_image_charge, dipole, conjugate_pair, fixed_charge, or fixed_dipole"
                )

        device = ctx.device
        dtype = ctx.dtype
        step = self.params["step"].value(device=device, dtype=dtype).reshape(-1)
        if step.numel() != 1:
            raise ValueError("ImageLadderNode param 'step' must be scalar")
        count_param = self.params["count"].value(device=device, dtype=dtype).reshape(-1)
        if count_param.numel() != 1:
            raise ValueError("ImageLadderNode param 'count' must be scalar")
        count_int = int(torch.round(count_param).clamp(min=1, max=256).item())
        decay_param = self.params.get("decay")
        if decay_param is None:
            decay_tensor = torch.as_tensor(0.0, device=device, dtype=dtype)
        else:
            decay_tensor = decay_param.value(device=device, dtype=dtype).reshape(-1)
        if decay_tensor.numel() != 1:
            raise ValueError("ImageLadderNode param 'decay' must be scalar if provided")
        decay = decay_tensor.view(1)[0]

        axis = (self.meta or {}).get("axis", "z")
        direction = int((self.meta or {}).get("direction", 1))
        axis_idx = _axis_index(axis)
        mode = self.meta.get("mode", "shared_slot")
        if mode != "shared_slot":
            raise NotImplementedError(
                f"ImageLadderNode mode '{mode}' not implemented (only shared_slot supported)"
            )

        group_policy = self.meta.get("group_policy", "inherit")
        motif_index = int(self.meta.get("motif_index", 0))

        child_contrib = child.lower(ctx)
        slots = _clone_slots_with_group_policy(
            child_contrib.slots, "image_ladder", group_policy, motif_index
        )
        child_K = child_contrib.evaluator.K
        alphas_template = torch.exp(-decay * torch.arange(count_int, device=device, dtype=dtype))

        def _alphas(X: torch.Tensor) -> torch.Tensor:
            return alphas_template.to(device=X.device, dtype=X.dtype)

        def _has_fixed(node: GFNode) -> bool:
            if isinstance(node, (FixedChargeNode, FixedDipoleNode)):
                return True
            if isinstance(node, SumNode):
                return any(_has_fixed(sub) for sub in node.children)
            return False

        def _aggregate_columns(node: GFNode, X: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
            if isinstance(node, RealImageChargeNode):
                pos = node.params["position"].value(device=device, dtype=dtype).reshape(1, 3)
                axis_vec = torch.zeros(3, device=pos.device, dtype=pos.dtype)
                axis_vec[axis_idx] = float(direction)
                offsets = (
                    torch.arange(count_int, device=pos.device, dtype=pos.dtype).view(-1, 1)
                    * step.view(1)
                    * axis_vec.view(1, 3)
                )
                shifted_pos = pos + offsets
                phi = coulomb_potential_real(X, shifted_pos)  # (N, count)
                weighted = phi * alphas.view(1, -1)
                return torch.sum(weighted, dim=1, keepdim=True)

            if isinstance(node, DipoleNode):
                pos = node.params["position"].value(device=device, dtype=dtype).reshape(1, 3)
                axis_vec = torch.zeros(3, device=pos.device, dtype=pos.dtype)
                axis_vec[axis_idx] = float(direction)
                offsets = (
                    torch.arange(count_int, device=pos.device, dtype=pos.dtype).view(-1, 1)
                    * step.view(1)
                    * axis_vec.view(1, 3)
                )
                shifted_pos = pos + offsets
                basis = dipole_basis_real(X, shifted_pos).reshape(X.shape[0], count_int, 3)
                weighted = basis * alphas.view(1, count_int, 1)
                return torch.sum(weighted, dim=1)

            if isinstance(node, ConjugatePairNode):
                complex_child = node.children[0]
                assert isinstance(complex_child, ComplexImageChargeNode)
                x = complex_child.params["x"].value(device=device, dtype=dtype).reshape(-1)
                y = complex_child.params["y"].value(device=device, dtype=dtype).reshape(-1)
                a = complex_child.params["a"].value(device=device, dtype=dtype).reshape(-1)
                b = complex_child.params["b"].value(device=device, dtype=dtype).reshape(-1)
                base = torch.stack((x, y, a, b), dim=-1).reshape(1, 4)
                axis_vec = torch.zeros(4, device=base.device, dtype=base.dtype)
                axis_vec[axis_idx] = float(direction)
                offsets = (
                    torch.arange(count_int, device=base.device, dtype=base.dtype).view(-1, 1)
                    * step.view(1)
                    * axis_vec.view(1, 4)
                )
                shifted = base + offsets
                cols = complex_conjugate_pair_columns(X, shifted).reshape(X.shape[0], count_int, 2)
                weighted = cols * alphas.view(1, count_int, 1)
                return torch.sum(weighted, dim=1)

            if isinstance(node, SumNode):
                parts = [_aggregate_columns(sub, X, alphas) for sub in node.children]
                parts = [p for p in parts if p.numel() > 0]
                return torch.cat(parts, dim=1) if parts else torch.zeros(X.shape[0], 0, device=X.device, dtype=X.dtype)

            if isinstance(node, (FixedChargeNode, FixedDipoleNode)):
                return torch.zeros(X.shape[0], 0, device=X.device, dtype=X.dtype)

            raise RuntimeError("Unsupported child type for ladder")

        def _aggregate_fixed(node: GFNode, X: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
            if isinstance(node, FixedChargeNode):
                pos = node.params["position"].value(device=device, dtype=dtype).reshape(1, 3)
                axis_vec = torch.zeros(3, device=pos.device, dtype=pos.dtype)
                axis_vec[axis_idx] = float(direction)
                offsets = (
                    torch.arange(count_int, device=pos.device, dtype=pos.dtype).view(-1, 1)
                    * step.view(1)
                    * axis_vec.view(1, 3)
                )
                shifted_pos = pos + offsets
                charge = node.params["charge"].value(device=device, dtype=dtype).reshape(1)
                phi = coulomb_potential_real(X, shifted_pos)  # (N, count)
                weighted = phi * alphas.view(1, -1)
                return torch.sum(weighted, dim=1) * charge.view(-1)[0]

            if isinstance(node, FixedDipoleNode):
                pos = node.params["position"].value(device=device, dtype=dtype).reshape(1, 3)
                axis_vec = torch.zeros(3, device=pos.device, dtype=pos.dtype)
                axis_vec[axis_idx] = float(direction)
                offsets = (
                    torch.arange(count_int, device=pos.device, dtype=pos.dtype).view(-1, 1)
                    * step.view(1)
                    * axis_vec.view(1, 3)
                )
                shifted_pos = pos + offsets
                moment = node.params["moment"].value(device=device, dtype=dtype).reshape(1, 3)
                basis = dipole_basis_real(X, shifted_pos).reshape(X.shape[0], count_int, 3)
                weighted_basis = basis * alphas.view(1, count_int, 1)
                summed = torch.sum(weighted_basis, dim=1)
                return torch.sum(summed * moment.view(1, 3), dim=1)

            if isinstance(node, SumNode):
                parts = [_aggregate_fixed(sub, X, alphas) for sub in node.children]
                return sum(parts) if parts else torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)

            return torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)

        def dense_fn(X: torch.Tensor) -> torch.Tensor:
            alphas = _alphas(X)
            ladder_cols = _aggregate_columns(child, X, alphas)
            if ladder_cols.dim() == 2 and ladder_cols.shape[1] == child_K:
                return ladder_cols
            return ladder_cols.reshape(X.shape[0], child_K)

        if ctx.eval_backend == "operator":

            def matvec_fn(w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
                alphas = _alphas(X)
                cols = _aggregate_columns(child, X, alphas)
                w_cast = w.to(device=X.device, dtype=X.dtype).reshape(-1)
                return cols @ w_cast

            def rmatvec_fn(r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
                alphas = _alphas(X)
                cols = _aggregate_columns(child, X, alphas)
                r_cast = r.to(device=X.device, dtype=X.dtype).reshape(-1)
                return cols.transpose(0, 1) @ r_cast

            evaluator = OperatorEvaluator(child_K, matvec_fn, rmatvec_fn, dense_fn=dense_fn)
        else:
            evaluator = DenseEvaluator(child_K, dense_fn)

        fixed_term_fn = None
        if _has_fixed(child) or child_contrib.fixed_term is not None:
            def fixed_term_fn(X: torch.Tensor) -> torch.Tensor:
                alphas = _alphas(X)
                return _aggregate_fixed(child, X, alphas)

        return LinearContribution(slots=slots, evaluator=evaluator, fixed_term=fixed_term_fn)


@register_node
@dataclass
class ConjugatePairNode(GFNode):
    node_type: ClassVar[str] = "conjugate_pair"

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.children) > 1:
            raise ValueError("ConjugatePairNode expects at most one child")

    def validate(self, ctx: Any) -> None:
        if len(self.children) != 1:
            raise ValueError("ConjugatePairNode must wrap exactly one child")

    def lower(self, ctx: CompileContext) -> LinearContribution:
        from electrodrive.gfdsl.compile.lower import (
            DenseEvaluator,
            LinearContribution,
            OperatorEvaluator,
        )
        from electrodrive.gfdsl.compile.types import CoeffSlot, CompileContext

        if len(self.children) != 1:
            raise ValueError("ConjugatePairNode must wrap exactly one child")
        child = self.children[0]
        if not isinstance(child, ComplexImageChargeNode):
            raise ValueError("ConjugatePairNode currently supports ComplexImageChargeNode child only")

        device = ctx.device
        dtype = ctx.dtype
        x = child.params["x"].value(device=device, dtype=dtype).reshape(-1)
        y = child.params["y"].value(device=device, dtype=dtype).reshape(-1)
        a = child.params["a"].value(device=device, dtype=dtype).reshape(-1)
        b = child.params["b"].value(device=device, dtype=dtype).reshape(-1)
        xyab = torch.stack((x, y, a, b), dim=-1).reshape(1, 4)

        base_hash = self.full_hash()
        group = self.group_info or child.group_info or GroupInfo(family_name="complex_pair")
        slots = [
            CoeffSlot(
                slot_id=f"{base_hash}:{idx}",
                dim=1,
                group_info=group,
                meta={"node_type": self.node_type, "component": comp},
            )
            for idx, comp in enumerate(("real", "imag"))
        ]

        def dense_fn(X: torch.Tensor) -> torch.Tensor:
            return complex_conjugate_pair_columns(X, xyab)

        if ctx.eval_backend == "operator":
            def matvec_fn(w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
                Phi = dense_fn(X)
                w_cast = w.to(device=X.device, dtype=X.dtype).reshape(-1)
                return Phi @ w_cast

            def rmatvec_fn(r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
                Phi = dense_fn(X)
                r_cast = r.to(device=X.device, dtype=X.dtype).reshape(-1)
                return Phi.transpose(0, 1) @ r_cast

            evaluator = OperatorEvaluator(2, matvec_fn, rmatvec_fn, dense_fn=dense_fn)
        else:
            evaluator = DenseEvaluator(2, dense_fn)

        return LinearContribution(slots=slots, evaluator=evaluator)


@register_node
@dataclass
class InterfaceLocalizedFamilyNode(GFNode):
    node_type: ClassVar[str] = "interface_localized_family"

    def validate(self, ctx: Any) -> None:
        if "scale" not in self.params:
            raise ValueError("InterfaceLocalizedFamilyNode requires param 'scale'")
        if "interface_id" not in (self.meta or {}):
            raise ValueError("InterfaceLocalizedFamilyNode requires meta['interface_id']")


@register_node
@dataclass
class DCIMBlockNode(GFNode):
    node_type: ClassVar[str] = "dcim_block"
    poles: Tuple[GFNode, ...] = field(default_factory=tuple)
    images: Tuple[GFNode, ...] = field(default_factory=tuple)
    branchcut: Optional[GFNode] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.poles = tuple(self.poles) if self.poles is not None else tuple()
        self.images = tuple(self.images) if self.images is not None else tuple()
        combined_children = list(self.poles) + list(self.images)
        if self.branchcut is not None:
            combined_children.append(self.branchcut)
        self.children = tuple(combined_children)

    def to_json_dict(self) -> Dict[str, Any]:
        data = super().to_json_dict()
        data["poles"] = [pole.to_json_dict() for pole in self.poles]
        data["images"] = [img.to_json_dict() for img in self.images]
        data["branchcut"] = (
            self.branchcut.to_json_dict() if self.branchcut is not None else None
        )
        return data

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "DCIMBlockNode":
        poles = _deserialize_children(data.get("poles", []))
        images = _deserialize_children(data.get("images", []))
        branchcut_raw = data.get("branchcut")
        branchcut = None
        if isinstance(branchcut_raw, dict):
            from .registry import make_node

            node_type = branchcut_raw.get("node_type")
            payload = {k: v for k, v in branchcut_raw.items() if k != "node_type"}
            branchcut = make_node(node_type, **payload)
        elif isinstance(branchcut_raw, GFNode):
            branchcut = branchcut_raw
        base_children = poles + images + (() if branchcut is None else (branchcut,))
        params = _deserialize_params(data.get("params", {}))
        meta = data.get("meta", {}) or {}
        group_info_raw = data.get("group_info")
        group_info = (
            GroupInfo.from_json_dict(group_info_raw)
            if group_info_raw is not None
            else None
        )
        return cls(
            poles=poles,
            images=images,
            branchcut=branchcut,
            children=base_children,
            params=params,
            meta=meta,
            group_info=group_info,
        )

    def validate(self, ctx: Any) -> None:
        if self.poles is None or self.images is None:
            raise ValueError("DCIMBlockNode requires 'poles' and 'images' fields")
        for child in list(self.poles) + list(self.images):
            child.validate(ctx)
        if self.branchcut is not None:
            self.branchcut.validate(ctx)

    def lower(self, ctx: CompileContext) -> LinearContribution:
        from electrodrive.gfdsl.compile.lower import (
            DenseEvaluator,
            LinearContribution,
            OperatorEvaluator,
        )

        contributions = []
        fixed_terms = []
        if self.poles:
            for pole in self.poles:
                contrib = pole.lower(ctx)
                contributions.append(contrib)
                if contrib.fixed_term is not None:
                    fixed_terms.append(contrib.fixed_term)
        if self.images:
            for img in self.images:
                contrib = img.lower(ctx)
                contributions.append(contrib)
                if contrib.fixed_term is not None:
                    fixed_terms.append(contrib.fixed_term)
        if self.branchcut is not None:
            contrib = self.branchcut.lower(ctx)
            contributions.append(contrib)
            if contrib.fixed_term is not None:
                fixed_terms.append(contrib.fixed_term)

        if not contributions:
            raise ValueError("DCIMBlockNode requires at least one lowered child")

        slots = []
        child_K: list[int] = []
        for contrib in contributions:
            slots.extend(contrib.slots)
            child_K.append(contrib.evaluator.K)
        total_K = sum(child_K)

        def dense_fn(X: torch.Tensor) -> torch.Tensor:
            if total_K == 0:
                return torch.zeros(X.shape[0], 0, device=X.device, dtype=X.dtype)
            cols = [
                contrib.evaluator.eval_columns(X)
                for contrib in contributions
                if contrib.evaluator.K > 0
            ]
            return torch.cat(cols, dim=1) if len(cols) > 1 else (cols[0] if cols else torch.zeros(X.shape[0], 0, device=X.device, dtype=X.dtype))

        if ctx.eval_backend == "operator":

            def matvec_fn(w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
                if total_K == 0:
                    return torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
                out = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
                offset = 0
                for contrib, k in zip(contributions, child_K):
                    w_slice = w[offset : offset + k]
                    if k > 0:
                        out = out + contrib.evaluator.matvec(w_slice, X)
                    offset += k
                return out

            def rmatvec_fn(r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
                if total_K == 0:
                    return torch.zeros(0, device=X.device, dtype=X.dtype)
                grads = [
                    contrib.evaluator.rmatvec(r, X)
                    for contrib in contributions
                    if contrib.evaluator.K > 0
                ]
                return torch.cat(grads, dim=0) if grads else torch.zeros(0, device=X.device, dtype=X.dtype)

            evaluator = OperatorEvaluator(total_K, matvec_fn, rmatvec_fn, dense_fn=dense_fn)
        else:
            evaluator = DenseEvaluator(total_K, dense_fn)

        fixed_term_fn = None
        if fixed_terms:
            def fixed_term_fn(X: torch.Tensor) -> torch.Tensor:
                return sum(fn(X) for fn in fixed_terms)

        return LinearContribution(slots=slots, evaluator=evaluator, fixed_term=fixed_term_fn)


@register_node
@dataclass
class SumNode(GFNode):
    node_type: ClassVar[str] = "sum"

    def validate(self, ctx: Any) -> None:
        for child in self.children:
            child.validate(ctx)

    def lower(self, ctx: CompileContext) -> LinearContribution:
        from electrodrive.gfdsl.compile.lower import (
            DenseEvaluator,
            LinearContribution,
            OperatorEvaluator,
        )
        from electrodrive.gfdsl.compile.types import CoeffSlot, CompileContext

        contributions = [child.lower(ctx) for child in self.children]
        slots = []
        child_K: list[int] = []
        fixed_terms = []
        for contrib in contributions:
            slots.extend(contrib.slots)
            child_K.append(contrib.evaluator.K)
            if contrib.fixed_term is not None:
                fixed_terms.append(contrib.fixed_term)

        total_K = sum(child_K)

        def _fixed_sum(X: torch.Tensor) -> torch.Tensor:
            if not fixed_terms:
                return torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
            return sum(fn(X) for fn in fixed_terms)

        def dense_fn(X: torch.Tensor) -> torch.Tensor:
            if total_K == 0:
                return torch.zeros(X.shape[0], 0, device=X.device, dtype=X.dtype)
            cols = [
                contrib.evaluator.eval_columns(X)
                for contrib in contributions
                if contrib.evaluator.K > 0
            ]
            return torch.cat(cols, dim=1) if len(cols) > 1 else (cols[0] if cols else torch.zeros(X.shape[0], 0, device=X.device, dtype=X.dtype))

        if ctx.eval_backend == "operator":
            def matvec_fn(w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
                out = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
                offset = 0
                for contrib, k in zip(contributions, child_K):
                    w_slice = w[offset : offset + k]
                    if k > 0:
                        out = out + contrib.evaluator.matvec(w_slice, X)
                    offset += k
                return out

            def rmatvec_fn(r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
                if total_K == 0:
                    return torch.zeros(0, device=X.device, dtype=X.dtype)
                grads = [
                    contrib.evaluator.rmatvec(r, X)
                    for contrib in contributions
                    if contrib.evaluator.K > 0
                ]
                return torch.cat(grads, dim=0)

            evaluator = OperatorEvaluator(total_K, matvec_fn, rmatvec_fn, dense_fn=dense_fn)
        else:
            evaluator = DenseEvaluator(total_K, dense_fn)

        fixed_term_fn = _fixed_sum if fixed_terms else None

        return LinearContribution(slots=slots, evaluator=evaluator, fixed_term=fixed_term_fn)


@register_node
@dataclass
class PiecewiseByRegionNode(GFNode):
    node_type: ClassVar[str] = "piecewise_by_region"

    def validate(self, ctx: Any) -> None:
        regions = self.meta.get("regions")
        if regions is None:
            raise ValueError("PiecewiseByRegionNode requires meta['regions']")
        if len(regions) != len(self.children):
            raise ValueError("PiecewiseByRegionNode regions count must match children")


@register_node
@dataclass
class LinearConstraintNode(GFNode):
    node_type: ClassVar[str] = "linear_constraint"

    def validate(self, ctx: Any) -> None:
        constraints = self.meta.get("constraints")
        if constraints is None:
            raise ValueError("LinearConstraintNode requires meta['constraints']")


@dataclass
class OpaqueNode(GFNode):
    """Placeholder for unknown/forward-compatible node types."""

    original_payload: Dict[str, Any] = field(default_factory=dict)
    original_node_type: Optional[str] = None
    node_type: ClassVar[str] = "opaque"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.original_node_type and "node_type" in self.original_payload:
            self.original_node_type = self.original_payload.get("node_type")

    def to_json_dict(self) -> Dict[str, Any]:
        payload = dict(self.original_payload) if self.original_payload else {}
        if "node_type" not in payload and self.original_node_type:
            payload["node_type"] = self.original_node_type
        if not payload:
            payload = super().to_json_dict()
            payload["node_type"] = self.original_node_type or payload.get("node_type")
        return payload

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "OpaqueNode":
        return cls(original_payload=data, original_node_type=data.get("node_type"))

    def validate(self, ctx: Any) -> None:
        return None
