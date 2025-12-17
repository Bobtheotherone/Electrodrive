"""State containers used during GFlowNet rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch

from electrodrive.gfn.dsl.canonicalization import hash_program
from electrodrive.gfn.dsl.nodes import Node
from electrodrive.gfn.dsl.program import Program
from electrodrive.gfn.dsl.tokenize import tokenize_program
from electrodrive.utils.device import get_default_device


_UNSET = object()


@dataclass(frozen=True)
class SpecMetadata:
    """Lightweight descriptor of the geometric specification."""

    geom_type: str
    n_dielectrics: int
    bc_type: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        """Return a serializable view of the spec metadata."""
        return {
            "geom_type": self.geom_type,
            "n_dielectrics": self.n_dielectrics,
            "bc_type": self.bc_type,
            "extra": dict(self.extra),
        }


@dataclass
class PartialProgramState:
    """State for a partial program during rollout."""

    spec_hash: str
    spec_meta: SpecMetadata
    ast_partial: Union[Program, Sequence[Node], None] = None
    action_mask: Optional[Tuple[bool, ...]] = None
    mask_cuda: Optional[torch.Tensor] = None
    mask_cache_key: Optional[Tuple[object, ...]] = None
    ast_token_ids: Optional[torch.Tensor] = None
    cached_embeddings: Any = None

    @property
    def program(self) -> Program:
        """Return the program backing this state."""
        if isinstance(self.ast_partial, Program):
            return self.ast_partial
        if self.ast_partial is None:
            return Program()
        return Program(nodes=tuple(self.ast_partial))

    @property
    def state_hash(self) -> str:
        """Stable identifier derived from the spec hash and canonical AST."""
        return hash_program(self.spec_hash, self.program.canonical_bytes)

    def with_action_mask(self, mask: Tuple[bool, ...]) -> "PartialProgramState":
        """Return a copy carrying an updated mask."""
        if self._is_frozen():
            return PartialProgramState(
                spec_hash=self.spec_hash,
                spec_meta=self.spec_meta,
                ast_partial=self.ast_partial,
                action_mask=mask,
                mask_cuda=None,
                mask_cache_key=None,
                ast_token_ids=self.ast_token_ids,
                cached_embeddings=self.cached_embeddings,
            )
        self.action_mask = mask
        self.mask_cuda = None
        self.mask_cache_key = None
        return self

    def with_tokens(self, token_ids: torch.Tensor) -> "PartialProgramState":
        """Return a copy carrying updated AST token ids."""
        if self._is_frozen():
            return PartialProgramState(
                spec_hash=self.spec_hash,
                spec_meta=self.spec_meta,
                ast_partial=self.ast_partial,
                action_mask=self.action_mask,
                mask_cuda=self.mask_cuda,
                mask_cache_key=self.mask_cache_key,
                ast_token_ids=token_ids,
                cached_embeddings=self.cached_embeddings,
            )
        self.ast_token_ids = token_ids
        return self

    def with_cached_mask(self, mask: torch.Tensor, key: Tuple[object, ...]) -> "PartialProgramState":
        """Cache a CUDA/CPU mask for this state."""
        if self._is_frozen():
            return PartialProgramState(
                spec_hash=self.spec_hash,
                spec_meta=self.spec_meta,
                ast_partial=self.ast_partial,
                action_mask=self.action_mask,
                mask_cuda=mask,
                mask_cache_key=key,
                ast_token_ids=self.ast_token_ids,
                cached_embeddings=self.cached_embeddings,
            )
        self.mask_cuda = mask
        self.mask_cache_key = key
        return self

    def copy_with(
        self,
        *,
        spec_hash: object = _UNSET,
        spec_meta: object = _UNSET,
        ast_partial: object = _UNSET,
        action_mask: object = _UNSET,
        mask_cuda: object = _UNSET,
        mask_cache_key: object = _UNSET,
        ast_token_ids: object = _UNSET,
        cached_embeddings: object = _UNSET,
    ) -> "PartialProgramState":
        """Return a copy with selected fields updated."""
        program_changed = ast_partial is not _UNSET
        spec_changed = spec_hash is not _UNSET or spec_meta is not _UNSET
        mask_changed = action_mask is not _UNSET

        if (program_changed or spec_changed or mask_changed) and mask_cuda is _UNSET:
            mask_cuda = None
        if (program_changed or spec_changed or mask_changed) and mask_cache_key is _UNSET:
            mask_cache_key = None

        new_state = PartialProgramState(
            spec_hash=self.spec_hash if spec_hash is _UNSET else spec_hash,
            spec_meta=self.spec_meta if spec_meta is _UNSET else spec_meta,
            ast_partial=self.ast_partial if ast_partial is _UNSET else ast_partial,
            action_mask=self.action_mask if action_mask is _UNSET else action_mask,
            mask_cuda=self.mask_cuda if mask_cuda is _UNSET else mask_cuda,
            mask_cache_key=self.mask_cache_key if mask_cache_key is _UNSET else mask_cache_key,
            ast_token_ids=self.ast_token_ids if ast_token_ids is _UNSET else ast_token_ids,
            cached_embeddings=self.cached_embeddings if cached_embeddings is _UNSET else cached_embeddings,
        )
        if self._is_frozen():
            return new_state
        self.__dict__.update(new_state.__dict__)
        return self

    def _is_frozen(self) -> bool:
        params = getattr(self, "__dataclass_params__", None)
        return bool(params and params.frozen)

    def to_token_sequence(self, max_len: int = 64) -> torch.Tensor:
        """Encode the AST into a fixed-length token sequence on the default device."""
        device = get_default_device()
        return tokenize_program(self.program, max_len=max_len, device=device)


__all__ = ["PartialProgramState", "SpecMetadata"]
