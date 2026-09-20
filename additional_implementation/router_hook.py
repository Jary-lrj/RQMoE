"""Hooks that isolate an intervention to the router's linear gate."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .metrics import RoutingDiagnostics


class RouterOnlyIntervention:
    """Temporarily replace only the input received by ``moe_layer.gate``."""

    def __init__(
        self,
        moe_layer: nn.Module,
        intervention: nn.Module,
        diagnostics: RoutingDiagnostics,
    ):
        gate = getattr(moe_layer, "gate", None)
        if not isinstance(gate, nn.Linear):
            raise TypeError("Expected moe_layer.gate to be torch.nn.Linear.")
        self.gate = gate
        self.intervention = intervention
        self.diagnostics = diagnostics
        self._pre_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._post_handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def _before_gate(self, _module: nn.Module, inputs: tuple) -> tuple:
        original = inputs[0]
        router_input = self.intervention(original)
        self.diagnostics.update_representation(original, router_input)
        return (router_input, *inputs[1:])

    def _after_gate(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        self.diagnostics.update_logits(output)

    def __enter__(self) -> "RouterOnlyIntervention":
        self._pre_handle = self.gate.register_forward_pre_hook(self._before_gate)
        self._post_handle = self.gate.register_forward_hook(self._after_gate)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._pre_handle is not None:
            self._pre_handle.remove()
        if self._post_handle is not None:
            self._post_handle.remove()
