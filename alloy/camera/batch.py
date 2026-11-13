# SPDX-License-Identifier: Apache-2.0
"""Render multiple compatible cameras in one full-material ray batch."""
from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import warp as wp

from .camera import Camera, scene_view, unwrap_sim
from .errors import CameraError
from .pbr.renderer import PbrRenderer
from .view import to_display


class PbrCameraBatch:
    """Reuse one ray batch for named cameras with matching look, size, samples and render settings."""

    def __init__(self, cameras: Mapping[str, Camera]):
        if not cameras:
            raise CameraError("an empty PBR camera batch", "pass at least one named Camera")
        self.cameras = dict(cameras)
        self._inputs = None
        self._renderer = None
        self._post_stream = self._display = self._host = None
        self._in_flight = False

    def render(self, source, *, state=None) -> dict[str, np.ndarray]:
        """Render the source's current state or an explicit snapshot with the same layout and device."""
        scene, renderer, first, spans = self._prepare(source, state)
        rgb, albedo, normal = renderer.render(first.spp)
        linear = first._denoised(rgb, albedo, normal, renderer)
        images = to_display(linear, view=scene_view(scene)).numpy()
        return dict(zip(self.cameras, (images[span] for span in spans)))

    def enqueue(self, source) -> Callable[[], dict[str, np.ndarray]]:
        """Trace now, then enqueue OptiX denoising and readback. Call the returned function before rendering again.

        The source may step while denoising runs on another CUDA stream. Finish on the calling thread to get the
        named RGB arrays; each is an owned snapshot of the traced state.
        """
        if any(camera.denoise.kind != "optix" for camera in self.cameras.values()):
            raise CameraError("asynchronous PBR batch needs OptiX denoising", "use denoise='optix'")
        scene, renderer, first, spans = self._prepare(source)
        device = wp.get_device(renderer.sc.device)
        if not device.is_cuda:
            raise CameraError("asynchronous PBR batch needs CUDA", "render the batch synchronously on CPU")
        rgb, albedo, normal = renderer.render(first.spp)
        shape = (*rgb.shape, 3)
        if self._post_stream is None or self._post_stream.device != device or self._host.shape != shape:
            self._post_stream = wp.Stream(device)
            self._display = wp.empty((rgb.size, 3), dtype=wp.uint8, device=device)
            self._host = wp.empty(shape, dtype=wp.uint8, device="cpu", pinned=True)
        stream = self._post_stream
        stream.wait_stream(wp.get_stream(device))
        try:
            with wp.ScopedStream(stream, sync_enter=False):
                linear = first._denoised(rgb, albedo, normal, renderer)
                display = to_display(linear, out=self._display, view=scene_view(scene))
                wp.copy(self._host, display, stream=stream)
        except BaseException:
            wp.synchronize_stream(stream)
            raise
        self._in_flight = True
        names = tuple(self.cameras)
        result = None

        def finish():
            nonlocal result
            if result is None:
                wp.synchronize_stream(stream)
                images = self._host.numpy().copy()
                result = dict(zip(names, (images[span] for span in spans)))
                self._in_flight = False
            return result

        return finish

    def _prepare(self, source, state=None):
        if self._in_flight:
            raise CameraError("asynchronous PBR batch is still rendering", "finish the pending frame before rendering again")
        _, state = unwrap_sim(source, state)
        prepared = set()
        cameras = list(self.cameras.values())
        first = cameras[0]
        inputs = [camera._prepare(source, state, prepared) for camera in cameras]
        scene, reference = inputs[0]
        for camera, (sc, renderer) in zip(cameras, inputs):
            if (not isinstance(renderer, PbrRenderer) or sc is not scene or renderer.st != reference.st
                    or (renderer.w, renderer.h, camera.spp, camera.denoise)
                    != (reference.w, reference.h, first.spp, first.denoise)):
                raise CameraError("incompatible cameras in a PBR batch",
                                  "give them the same full-material look, size, samples, denoiser and settings")
        if self._inputs != inputs:
            worlds = np.concatenate([renderer.worlds for _, renderer in inputs])
            self._renderer = PbrRenderer(scene, reference.w, reference.h, reference.st, worlds=worlds)
            self._inputs = inputs
        renderer = self._renderer
        offset = 0
        spans = []
        for _, view in inputs:
            for field in ("eye", "rot", "extent", "clip", "aperture", "focus"):
                wp.copy(getattr(renderer.cams, field), getattr(view.cams, field),
                        dest_offset=offset, count=view.n_img)
            spans.append(slice(offset, offset + view.n_img))
            offset += view.n_img
        return scene, renderer, first, spans
