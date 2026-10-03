"""DataExtractor base class -- extensibility core."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DataExtractor(ABC):
    """Each subclass calls one small-model service to extract one kind of data.

    To add a new data source:
      1. Subclass this class
      2. Declare ``name`` and ``depends_on``
      3. Implement ``extract()``
      4. Register in DetectionRunner
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique key under context.raw_data."""
        ...

    @property
    def depends_on(self) -> list[str]:
        """Names of extractors that must run before this one."""
        return []

    @abstractmethod
    def extract(self, image_b64: str, prior: dict[str, Any]) -> Any:
        """Extract data.  *prior* holds outputs of dependency extractors."""
        ...
