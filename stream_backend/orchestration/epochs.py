from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class EpochClock:
    generated_at: str

    @property
    def date_label(self) -> str:
        return self.generated_at.split("T", 1)[0]

    @property
    def run_label(self) -> str:
        return (
            self.generated_at.replace("-", "")
            .replace(":", "")
            .replace("+00:00", "z")
            .replace(".", "")
            .replace("T", "t")
        )

    @classmethod
    def now(cls) -> "EpochClock":
        value = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        return cls(generated_at=value)
