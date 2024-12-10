import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

src = glob.glob("./cuda/**/*.cu") + glob.glob("./cpu/**/*.cpp") + ["./interface/RadonTorch.cpp"]

setup(
    name='deeprecon',
    version='2.0.0',
    author='Wenjun Xia',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
        name = 'radontorch',
        sources = src,
        include_dirs=[
            "./include/dis2d",
            "./include/dis3d",
            os.path.abspath('include')
        ],),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
