"""Electrodrive Green's Function DSL (GFDSL).

This package is experimental and not wired into the default discovery pipeline.
Use the compatibility adapter (to be added in later milestones) to opt in.
"""

from electrodrive.gfdsl import ast, compile, eval, io

__all__ = ["ast", "compile", "eval", "io"]
