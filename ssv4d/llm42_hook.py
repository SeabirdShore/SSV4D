"""Optional LLM-42 verification gate adapter.

The hook preserves LLM-42's default behavior unless a gate is installed. It is
deliberately fail-open: exceptions or missing risk state cause verification.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional


Gate = Callable[[Any], bool]


def install_verification_gate(worker: Any, gate: Gate) -> None:
    """Install a per-request gate on a live ``LLM42Worker`` instance.

    Denied requests are temporarily marked non-deterministic only for the
    duration of ``check_and_verify_deterministic_requests``. This skips the
    verifier without changing the draft decode or the request's long-term
    deterministic policy.
    """

    if getattr(worker, "_ssv4d_gate_installed", False):
        return

    original = worker.check_and_verify_deterministic_requests

    @functools.wraps(original)
    def gated(batch):
        denied: list[Any] = []
        if batch is not None and getattr(batch, "reqs", None):
            for req in batch.reqs:
                if not getattr(req, "is_deterministic", False):
                    continue
                try:
                    should_verify = bool(gate(req))
                except Exception:
                    should_verify = True
                if not should_verify:
                    denied.append(req)
                    req.is_deterministic = False
        try:
            return original(batch)
        finally:
            for req in denied:
                req.is_deterministic = True

    worker.check_and_verify_deterministic_requests = gated
    worker._ssv4d_gate_installed = True
    worker._ssv4d_original_check = original


def install_class_gate(worker_cls: Any, gate: Gate) -> None:
    """Install the gate on all future instances of ``worker_cls``."""

    original = worker_cls.check_and_verify_deterministic_requests

    @functools.wraps(original)
    def gated(self, batch):
        if not getattr(self, "_ssv4d_gate_installed", False):
            install_verification_gate(self, gate)
        return self.check_and_verify_deterministic_requests(batch)

    worker_cls.check_and_verify_deterministic_requests = gated
