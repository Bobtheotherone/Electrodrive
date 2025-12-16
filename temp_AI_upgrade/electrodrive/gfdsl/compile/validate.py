"""Validation helpers for GFDSL programs."""

from __future__ import annotations

from electrodrive.gfdsl.ast.nodes import DCIMBlockNode, GFNode
from electrodrive.gfdsl.compile.types import CompileContext


class GFDSLValidationError(Exception):
    def __init__(self, path: str, message: str):
        super().__init__(f"Validation failed at {path}: {message}")
        self.path = path
        self.message = message


def _validate_node(node: GFNode, ctx: CompileContext, path: str) -> None:
    try:
        node.validate(ctx)
    except Exception as exc:
        raise GFDSLValidationError(path, str(exc)) from exc

    for idx, child in enumerate(node.children):
        _validate_node(child, ctx, f"{path}.children[{idx}]")

    if isinstance(node, DCIMBlockNode):
        for idx, pole in enumerate(node.poles):
            _validate_node(pole, ctx, f"{path}.poles[{idx}]")
        for idx, image in enumerate(node.images):
            _validate_node(image, ctx, f"{path}.images[{idx}]")
        if node.branchcut is not None:
            _validate_node(node.branchcut, ctx, f"{path}.branchcut")


def validate_program(root: GFNode, ctx: CompileContext | None = None) -> None:
    """Validate a GFDSL program, raising GFDSLValidationError on failure."""
    ctx = ctx or CompileContext()
    _validate_node(root, ctx, path="root")
