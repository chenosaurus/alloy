# SPDX-License-Identifier: Apache-2.0
"""OptiX HDR denoising of linear RGB with albedo and normal guides on Warp's stream."""
from __future__ import annotations

import warp as wp

from ..errors import CameraError
from .lib import check, lib


class OptixDenoiser:
    def __init__(self, width: int, height: int, device: str):
        self.device = wp.get_device(device)
        self.width, self.height = width, height
        self._stream = wp.get_stream(self.device)
        self._event = wp.Event(self.device)
        self._out = None
        self._handle = lib().owf_denoiser_create(self.device.context, self._stream.cuda_stream, width, height)
        if not self._handle:
            check(-1, "denoiser setup")
        self._stream.record_event(self._event)

    def __call__(self, rgb: wp.array, albedo: wp.array, normal: wp.array, spp: int) -> wp.array:
        n = rgb.shape[0]
        shape = (n, self.height, self.width)
        for guide in (rgb, albedo, normal):
            if (guide.shape != shape or guide.dtype != wp.vec3 or guide.device != self.device
                    or not guide.is_contiguous):
                raise CameraError("invalid OptiX denoiser input", f"pass contiguous vec3 arrays of shape {shape}")
        if not self._handle:
            raise CameraError("the OptiX denoiser is closed", "create a new denoiser")
        stream = wp.get_stream(self.device)
        if stream != self._stream:
            stream.wait_event(self._event)
            self._stream = stream
        if self._out is None or self._out.shape != shape:
            self._out = wp.empty(shape, dtype=wp.vec3, device=self.device)
        stride = self.width * self.height * 12
        try:
            for i in range(n):
                offset = i * stride
                check(lib().owf_denoiser_invoke(self._handle, stream.cuda_stream, rgb.ptr + offset,
                                                albedo.ptr + offset, normal.ptr + offset, self._out.ptr + offset),
                      "denoising")
        finally:
            stream.record_event(self._event)
        return self._out

    def close(self) -> None:
        if getattr(self, "_handle", None):
            wp.synchronize_stream(self._stream)
            lib().owf_denoiser_destroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # interpreter shutdown
            pass
