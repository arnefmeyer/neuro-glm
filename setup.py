
from setuptools import setup, find_packages

setup(
    name='neuro_glm',
    version='0.1',
    url='https://github.com/arnefmeyer/neuro_glm.git',
    author='Arne Meyer',
    author_email='arne.f.meyer@gmail.com',
    description='Simple Poisson GLM for neural data analysis',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'scipy'],
)
