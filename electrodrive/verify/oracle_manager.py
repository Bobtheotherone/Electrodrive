from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from .oracle_registry import OracleBackend, OracleRegistry
from .oracle_types import OracleFidelity, OracleQuery, OracleResult


class OracleManager:
    def __init__(self, registry: OracleRegistry) -> None:
        self._registry = registry
        self._last_backend: Optional[OracleBackend] = None
        self._last_failures: List[Tuple[str, str]] = []

    def evaluate(self, query: OracleQuery) -> OracleResult:
        result, _ = self.evaluate_with_backend(query)
        return result

    def evaluate_with_backend(self, query: OracleQuery) -> tuple[OracleResult, OracleBackend]:
        failures: List[Tuple[str, str]] = []
        for backend in self._iter_backends(query):
            self._last_backend = backend
            try:
                result = backend.evaluate(query)
                self._last_failures = failures
                return result, backend
            except Exception as exc:
                failures.append((backend.name, str(exc)))
                continue
        self._last_failures = failures
        if failures:
            detail = "; ".join([f"{name}: {err}" for name, err in failures])
            raise RuntimeError(f"All {len(failures)} oracle backends failed: {detail}")
        raise RuntimeError("No oracle backend available for query")

    def _resolve_fidelity_chain(self, query: OracleQuery) -> Iterable[OracleFidelity]:
        if query.requested_fidelity != OracleFidelity.AUTO:
            return [query.requested_fidelity]
        return [
            OracleFidelity.F0,
            OracleFidelity.F1,
            OracleFidelity.F2,
            OracleFidelity.F3,
        ]

    def _iter_backends(self, query: OracleQuery) -> Iterable[OracleBackend]:
        if query.requested_fidelity == OracleFidelity.AUTO:
            for backend in self._registry.select(OracleFidelity.F0, query):
                yield backend
            allow_f1 = True
            if isinstance(query.budget, dict):
                allow_f1 = bool(query.budget.get("allow_f1_auto", True))
            if allow_f1:
                for backend in self._registry.select(OracleFidelity.F1, query):
                    yield backend
            allow_f2 = False
            if isinstance(query.budget, dict):
                allow_f2 = bool(query.budget.get("allow_f2_auto", False))
            if allow_f2:
                backend = self._registry.select_one(OracleFidelity.F2, query)
                if backend is not None:
                    yield backend
            return
        for backend in self._registry.select(query.requested_fidelity, query):
            yield backend

    def list_candidates(self, query: OracleQuery) -> List[OracleBackend]:
        backends: List[OracleBackend] = []
        for backend in self._iter_backends(query):
            backends.append(backend)
        return backends

    def should_escalate(self, _result: OracleResult) -> bool:
        return False

    def maybe_escalate(self, _result: OracleResult) -> Optional[OracleFidelity]:
        return None

    def escalate(self, _query: OracleQuery, _context: Optional[dict] = None) -> Optional[OracleFidelity]:
        return None

    def get_last_backend(self) -> Optional[OracleBackend]:
        return self._last_backend
