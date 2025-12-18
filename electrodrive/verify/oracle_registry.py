from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from .oracle_types import OracleFidelity, OracleQuery, OracleResult


class OracleBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def fidelity(self) -> OracleFidelity:
        raise NotImplementedError

    @abstractmethod
    def can_handle(self, query: OracleQuery) -> bool:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, query: OracleQuery) -> OracleResult:
        raise NotImplementedError

    @abstractmethod
    def fingerprint(self) -> str:
        raise NotImplementedError


class OracleRegistry:
    def __init__(self) -> None:
        self._backends: List[OracleBackend] = []

    def register(self, backend: OracleBackend) -> None:
        self._backends.append(backend)

    def list_backends(self) -> List[OracleBackend]:
        return list(self._backends)

    def select(self, fidelity: OracleFidelity, query: OracleQuery) -> List[OracleBackend]:
        return [
            backend
            for backend in self._backends
            if backend.fidelity == fidelity and backend.can_handle(query)
        ]

    def select_one(self, fidelity: OracleFidelity, query: OracleQuery) -> Optional[OracleBackend]:
        matches = self.select(fidelity, query)
        return matches[0] if matches else None
