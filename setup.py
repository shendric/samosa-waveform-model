from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension(
        name="samosa_waveform_model.model_cy",
        sources=["src/samosa_waveform_model/model_cy.pyx"],
        include_dirs=[np.get_include()],
    ),
]


setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
