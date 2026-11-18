"""Compositing shared by the dense and sparse smoke ray marchers."""

import taichi as ti


@ti.func
def composite_sample(color, transmittance, sample_color, light, opacity):
    transmittance *= 1.0 - opacity
    color += opacity * sample_color * light * transmittance
    return color, transmittance
