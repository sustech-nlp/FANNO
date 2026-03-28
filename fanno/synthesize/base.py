"""Base synthesizer abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from loguru import logger


class BaseSynthesizer(ABC):
    """Base class for all FANNO data synthesizers.

    Subclasses implement generate() and validate() for specific data types
    (QA, creative, dialog, agent).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        workers: int = 8,
    ) -> None:
        self.model = model
        self.workers = workers
        self._api_client = None

    @property
    def api_client(self):
        """Lazy-initialize Azure API client."""
        if self._api_client is None:
            from fanno.api.client import AzureAPIClient
            self._api_client = AzureAPIClient(
                model_name=self.model,
                workers=self.workers,
            )
        return self._api_client

    @abstractmethod
    def generate(self, num_samples: int, **kwargs) -> List[Dict[str, Any]]:
        """Generate synthetic data samples."""
        ...

    @abstractmethod
    def validate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and filter generated data."""
        ...

    def format_output(
        self,
        data: List[Dict[str, Any]],
        fmt: str = "alpaca",
    ) -> List[Dict[str, Any]]:
        """Convert to target format."""
        from fanno.data.formats import to_alpaca_format, to_sharegpt_format, to_agent_format
        if fmt == "alpaca":
            return to_alpaca_format(data)
        if fmt == "sharegpt":
            return to_sharegpt_format(data)
        if fmt == "agent":
            return to_agent_format(data)
        return data

    def generate_and_validate(
        self,
        num_samples: int,
        max_attempts: int = 3,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate and validate data with retries for insufficient yield.

        Args:
            num_samples: Target number of valid samples.
            max_attempts: Maximum generation rounds.

        Returns:
            List of validated data samples.
        """
        all_valid: List[Dict[str, Any]] = []

        for attempt in range(max_attempts):
            remaining = num_samples - len(all_valid)
            if remaining <= 0:
                break

            # Generate more than needed to account for filtering
            batch_size = int(remaining * 1.3) + 10
            logger.info(
                f"Generation attempt {attempt + 1}/{max_attempts}: "
                f"generating {batch_size} samples (need {remaining} more)"
            )
            raw = self.generate(batch_size, **kwargs)
            valid = self.validate(raw)
            all_valid.extend(valid)
            logger.info(f"Validated {len(valid)}/{len(raw)} samples")

        return all_valid[:num_samples]


__all__ = ["BaseSynthesizer"]
