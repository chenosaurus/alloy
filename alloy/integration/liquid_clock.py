# SPDX-License-Identifier: Apache-2.0
"""Opt-in settings for interactive GPU scenes with rigid bodies and SPH water."""
from __future__ import annotations

import contextlib
import contextvars
import dataclasses
from collections.abc import Iterator

from alloy.core.errors import CouplingError
from alloy.coupling.types import PairPolicy, SchedulerPolicy
from alloy.scene.types import BuiltScene, EntrySpec
from alloy.core.timebase import snap_to_grid

_INTERACTIVE = contextvars.ContextVar("alloy_liquid_interactive", default=False)
_EXCHANGE = .005
_RIGID_DT = .0025


@contextlib.contextmanager
def requested(interactive: bool) -> Iterator[None]:
    token = _INTERACTIVE.set(interactive)
    try:
        yield
    finally:
        _INTERACTIVE.reset(token)


def interactive() -> bool:
    return _INTERACTIVE.get()


def configure(built: BuiltScene, native: str, entries: list[EntrySpec], pairs: tuple[PairPolicy, ...],
              policy: SchedulerPolicy) -> tuple[list[EntrySpec], tuple[PairPolicy, ...], SchedulerPolicy]:
    """Use one lagged exchange while retaining two-way forces and impact refinement."""
    if (not built.model.device.is_cuda or native != "mujoco" or built.precision_pairs
            or built.coupler.kind != "proxy" or {e.capability for e in entries} != {"mujoco", "sph"}
            or len(pairs) != 1 or pairs[0].one_way):
        raise CouplingError("interactive liquid coupling needs a CUDA MuJoCo and SPH pair with native rigid contacts")
    nominal = snap_to_grid(_EXCHANGE, built.control_dt)
    configured = []
    for entry in entries:
        if entry.capability == "sph":
            # More velocity smoothing keeps resting water quiet with the demo's coarse pressure solve.
            kwargs = dict(entry.solver_kwargs, dt_max=_EXCHANGE, rest_jitter_speed=None,
                          divergence_solve=False, max_iterations=12, xsph_c=.15)
            # Start at one substep and let the existing CFL estimator refine during fast flow.
            configured.append(dataclasses.replace(entry, substeps=1, max_substeps=max(policy.clock.substep_ladder),
                                                  solver_kwargs=kwargs))
        else:
            rigid_rung = 2 if nominal > _RIGID_DT else 1
            configured.append(dataclasses.replace(entry, substeps=rigid_rung, max_substeps=1))
    return configured, (dataclasses.replace(pairs[0], mode="lagged"),), dataclasses.replace(
        policy, lagged_iterations=1,
        clock=dataclasses.replace(policy.clock, dt_x_max=_EXCHANGE, keep_rigid_step_on_impact=True),
        watchdog=dataclasses.replace(policy.watchdog, predict_onset_level=True),
    )
