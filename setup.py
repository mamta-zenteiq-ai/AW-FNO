from setuptools import setup, find_packages

setup(
    name="awfno",
    version="0.1.0",
    description="Adaptive Wavelet-Fourier Neural Operator",
    author="Mamta",
    packages=find_packages(exclude=["scripts", "scripts.*", "data", "data.*"]),
    install_requires=[
        "torch",
        "numpy",
        "scipy",
        "PyWavelets",
        "pyyaml",
        "matplotlib",
    ],
)
