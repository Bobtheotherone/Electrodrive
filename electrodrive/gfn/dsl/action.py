"""Typed action tokens used by factorized GFlowNet policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

ActionToken = Tuple[str, Optional[str], Tuple[Tuple[str, Any], ...], Optional[Tuple[Tuple[str, float], ...]]]


@dataclass(frozen=True)
class Action:
    """Structured action description suitable for factorized heads."""

    action_type: str
    action_subtype: Optional[str] = None
    discrete_args: Mapping[str, Any] = field(default_factory=dict)
    continuous_args: Optional[Mapping[str, float]] = None

    def to_token(self) -> ActionToken:
        """Serialize the action into a deterministic tuple token."""
        discrete = tuple(sorted(self.discrete_args.items()))
        continuous: Optional[Tuple[Tuple[str, float], ...]] = None
        if self.continuous_args:
            continuous = tuple(sorted(self.continuous_args.items()))
        return (self.action_type, self.action_subtype, discrete, continuous)

    @classmethod
    def from_token(cls, token: ActionToken) -> "Action":
        """Deserialize an action token produced by :meth:`to_token`."""
        action_type, action_subtype, discrete, continuous = token
        discrete_args = {k: v for k, v in discrete}
        continuous_args = {k: v for k, v in continuous} if continuous is not None else None
        return cls(
            action_type=action_type,
            action_subtype=action_subtype,
            discrete_args=discrete_args,
            continuous_args=continuous_args,
        )

    def as_dict(self) -> Mapping[str, Any]:
        """Return a JSON-friendly view of the action."""
        return {
            "action_type": self.action_type,
            "action_subtype": self.action_subtype,
            "discrete_args": dict(self.discrete_args),
            "continuous_args": dict(self.continuous_args) if self.continuous_args else None,
        }

    @property
    def token_length(self) -> int:
        """Length of the token tuple; useful for policy head bookkeeping."""
        return len(self.to_token())


__all__ = ["Action", "ActionToken"]
