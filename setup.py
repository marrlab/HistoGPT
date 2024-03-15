from setuptools import setup, find_packages

setup(
    name="histogpt",
    version="0.1.0",
    description="HistoGPT - PyTorch",
    packages=find_packages(exclude=[]),
    install_requires=[
        "einops>=0.4",
        "einops-exts",
        #"openslide-python>=1.3.1",
        "sacremoses>=0.1.1",
        "slideio>=2.5.0",
        "torch>=2.1.0",
        "transformers>=4.37.2",
        #"flamingo-pytorch>=0.1.2",
    ],
)
