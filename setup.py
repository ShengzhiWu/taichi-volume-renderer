import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="taichi-volume-renderer",
    version="1.3.0",
    author="Shengzhi Wu",
    author_email="e1124755@u.nus.edu",
    description="A python package for real-time GPU volume rendering based on taichi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShengzhiWu/taichi-volume-renderer",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'taichi'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Visualization"
    ],
    keywords="taichi volume rendering 3d visualization"
)