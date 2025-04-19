# taichi-volume-renderer
taichi-volume-renderer is a python package for real-time GPU volume rendering based on [taichi](https://github.com/taichi-dev/taichi).

```bash
pip install taichi-volume-renderer
```

You don't need to understand Taichi to use this package. For the simplest application — visualizing a 3D scalar NumPy array `a` as volume smoke — you can do it with just one line of code:

```python
import taichi_volume_renderer

taichi_volume_renderer.plot_volume(a)
```

## Usage

Example see `examples/example.py`.

![0](/images/0.png)
