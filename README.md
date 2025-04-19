# taichi-volume-renderer
taichi-volume-renderer is a python package for real-time GPU volume rendering based on [taichi](https://github.com/taichi-dev/taichi).

You don't need to understand Taichi to use this package. For the simplest application — visualizing a 3D scalar NumPy array `a` as volume smoke — you can do it with just one line of code:

```python
import taichi_volume_renderer

taichi_volume_renderer.plot_volume(a)
```

## Installation

```bash
pip install taichi-volume-renderer
```

## Usage

The simplest example would be rendering a static scene, with smoke density, color, and lighting all specified by a few NumPy arrays. See `examples/example.py`.

![0](/images/0.png)

The taichi-volume-renderer is built to work flawlessly with Taichi, enabling dynamic scene visualization. The following example solves a partial differential equation (PDE), specifically the Gray-Scott model, while visualizing the system's evolution in real-time. See `examples/pde.py`.

![1](/images/1.png)
