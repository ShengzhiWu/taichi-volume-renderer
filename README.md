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

### Interactive Static Scenes

The simplest example would be rendering a static scene, with smoke density, color, and lighting all specified by a few NumPy arrays. See `examples/example.py`.

![0](/images/0.jpg)

Volume rendering provides an impressive capability to display faintly visible objects with indistinct boundaries. The following example visualizes a Lorenz attractor. See `examples/strange_attractor.py`.

![2](/images/lorenz-attractor.jpg)

### High-Performance Real-Time Visualization

The taichi-volume-renderer is built to work flawlessly with Taichi, enabling dynamic scene visualization. The following example solves a partial differential equation (PDE), specifically the Gray-Scott model, while visualizing the system's evolution in real-time. The script also saves an `.gif` animation. See `examples/pde.py`.

![1](/images/pde.gif)

I also made a video demonstrating the dazzlingly complex behavior of this system through parameter changes. Check it out at https://www.bilibili.com/video/BV1g7LVzVEQW/

### VDB

You can also render VDB data with taichi-volume-renderer. See `examples/openvdb.py`. We can apply general lighting, or illuminate the volume from within like cloud-to-cloud lightning.

![cloud](images/cloud.jpg)

### Refraction

Refractive volume rendering allows visualizing objects with refractive behavior without constructing a mesh. See `examples/refraction.py`, where a glass ball is rendered.

![refraction](images/refraction.png)

The following example uses the Position Based Fluids (PBF) algorithm (Macklin, M. and Müller, M., 2013) to simulate fluids in real-time and renders them as a transparent material. The code is adapted from Ye Kuang's Taichi demo, [pbf2d.py](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/pbf2d.py). Since no meshing is required, the entire pipeline can be executed on the GPU. See `examples/pbf3d.py`.

![pbf3d](images/pbf3d.gif)

Continuous distribution of refractive indices is supported, allowing simulating phenomena such as heat haze or mirages.

## TODO

1. Default lights
2. Background images
3. Ray intersection with the scene
4. Reflection
5. Secondary scattering
6. Sparse grid
