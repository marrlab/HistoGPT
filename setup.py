from setuptools import setup, find_packages

setup(
    name="histogpt",
    version="2.0.0",
    description="HistoGPT - PyTorch",
    packages=find_packages(exclude=[]),
    install_requires=[
        "einops>=0.7.0",
        "einops-exts>=0.0.4",
        "flash-attn>=2.3.0",
        "flash-perceiver>=0.2.0",
        #"flamingo-pytorch>=0.1.2",
        "openai>=1.32.0",
        #"openslide-python>=1.3.1",
        "sacremoses>=0.0.53",
        "slideio>=2.2.0",
        "timm>=0.9.16",
        "torch>=2.1.0",
        "transformers>=4.37.2",
    ],
)
