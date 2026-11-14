"""Pre-prefill batch perturbation plans for the shadow branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PerturbationPlan:
    name: str
    batch_size: int
    protected_index: int
    prompts: tuple[str, ...]
    family: str
    repeat: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "batch_size": self.batch_size,
            "protected_index": self.protected_index,
            "family": self.family,
            "repeat": self.repeat,
            "prompts": list(self.prompts),
        }


def _place(target: str, fillers: list[str], position: str) -> tuple[list[str], int]:
    if position == "first":
        prompts = [target, *fillers]
        index = 0
    elif position == "middle":
        split = len(fillers) // 2
        prompts = [*fillers[:split], target, *fillers[split:]]
        index = split
    elif position == "last":
        prompts = [*fillers, target]
        index = len(fillers)
    else:
        raise ValueError(f"Unknown position: {position}")
    return prompts, index


def make_perturbation_plans(
    protected_prompt: str,
    fillers: Iterable[str],
    *,
    batch_sizes: tuple[int, ...] = (4, 8, 16),
    positions: tuple[str, ...] = ("first", "middle", "last"),
    repeats: int = 1,
) -> list[PerturbationPlan]:
    """Create deterministic, reproducible batch-shape perturbations.

    The protected prompt is inserted among different filler prompts and at
    different positions. Repeated plans rotate the filler window, producing a
    second batch-order perturbation without changing vocabulary.
    """

    filler_list = list(fillers)
    if not filler_list:
        raise ValueError("At least one filler prompt is required")

    plans: list[PerturbationPlan] = [
        PerturbationPlan(
            name="bs1",
            batch_size=1,
            protected_index=0,
            prompts=(protected_prompt,),
            family="batch_size",
        )
    ]

    for size in batch_sizes:
        if size < 2:
            continue
        for repeat in range(repeats):
            offset = (repeat * max(1, size // 2)) % len(filler_list)
            rotated = filler_list[offset:] + filler_list[:offset]
            needed = size - 1
            selected = (rotated * ((needed // len(rotated)) + 1))[:needed]
            for position in positions:
                prompts, protected_index = _place(
                    protected_prompt, selected, position
                )
                plans.append(
                    PerturbationPlan(
                        name=f"bs{size}_{position}_r{repeat}",
                        batch_size=size,
                        protected_index=protected_index,
                        prompts=tuple(prompts),
                        family=f"batch_size+order+{position}",
                        repeat=repeat,
                    )
                )
    return plans
