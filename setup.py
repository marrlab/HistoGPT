import setuptools

setuptools.setup(
    name="histogpt",
    version="0.1.0",
    description="HistoGPT - PyTorch",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.37.2",
        "flamingo-pytorch>=0.1.2",
    ],
)
