"""Utilities for validating the LLM-42 verified commit frontier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FrontierSnapshot:
    rid: str
    verified_tokens: int
    output_tokens: int
    unverified_tokens: int
    finished: bool

    @property
    def fully_verified(self) -> bool:
        return self.verified_tokens == self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def frontier_snapshot(req: Any) -> FrontierSnapshot:
    verified = int(req.llm42_verified_tokens)
    output = len(req.output_ids)
    return FrontierSnapshot(
        rid=req.rid,
        verified_tokens=verified,
        output_tokens=output,
        unverified_tokens=max(0, output - verified),
        finished=req.finished_reason is not None,
    )


def validate_frontier(snapshot: FrontierSnapshot) -> None:
    if snapshot.verified_tokens < 0:
        raise ValueError(f"Negative verified frontier: {snapshot}")
    if snapshot.verified_tokens > snapshot.output_tokens:
        raise ValueError(
            f"Verified frontier exceeds output length: {snapshot}"
        )
    if snapshot.finished and not snapshot.fully_verified:
        raise ValueError(
            f"Finished request has unverified speculative tokens: {snapshot}"
        )
