
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import find_packages, setup
from setuptools.extension import Extension
import numpy as np

extensions = [
    Extension("samosa_waveform_model.funcs", ["src/samosa_waveform_model/funcs.pyx"])
]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}
    ),
    include_dirs=[np.get_include()]
)