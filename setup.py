from setuptools import setup, find_packages

setup(
    name="histogpt",
    version="1.1.2",
    description="HistoGPT - PyTorch",
    packages=find_packages(exclude=[]),
    install_requires=[
        "einops>=0.4",
        "einops-exts",
        "openai>=1.14.0",
        #"openslide-python>=1.3.1",
        "sacremoses>=0.1.1",
        "slideio>=2.7.1",
        "torch>=2.1.0",
        "transformers==4.38.2",
        #"flamingo-pytorch>=0.1.2",
    ],
)
