# HistoGPT

[[preprint](https://www.medrxiv.org/content/10.1101/2024.03.15.24304211v2)] [[weights](https://huggingface.co/marr-peng-lab/histogpt)] [[notebook](https://github.com/marrlab/HistoGPT/blob/main/tutorial-2.ipynb)]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marrlab/HistoGPT/blob/main/tutorial-2.ipynb)

## Generating clinical-grade pathology reports from whole slide images
HistoGPT is a vision language foundation model for dermatopathology. The model takes a series of tissue sections from the same patient as input and generates a pathology report that includes the disease classification, tumor subtype prediction, tumor thickness estimation, and other important clinical information. Most importantly, HistoGPT is fully interpretable, as every word or phrase in the output text can be visualized in the original image.

<img src="github/figure-1.png" width="500"/>

We trained HistoGPT on a large-scale dataset of 6,705 patient-report pairs from over 15,000 whole slide images (WSIs) of over 150 different skin conditions (healthy, inflammatory, cancerous, ...) provided by the Department of Dermatology at the Technical University of Munich (TUM). To test our model, we extensively evaluated HistoGPT on five external cohorts from five different countries, including a dataset of 1,300 patient-report pairs from the Department of Dermatology at the University Hospital Münster (UKM).

## HistoGPT simultaneously learns from vision and language
HistoGPT takes a series of whole slide images as input, crops them into smaller image patches, extracts the feature vectors for each image patch with an image encoder (Uni), encodes the position information with a three-dimensional factorized position embedder (NaViT), downsamples them to a fixed number of latent vectors with a slide encoder (Perceiver Resampler), and combines them with text features from a language model (BioGPT) via interleaved tanh-gated cross-attention layers (XATTN).

<img src="github/figure-2.png" width="790"/>

## HistoGPT is simple and easy to use

We can install HistoGPT with the following commands
```
pip install flamingo-pytorch --no-deps
pip install git+https://github.com/marrlab/HistoGPT
```

For visualization, we also need
```
pip install openslide-python
```

The forward pass of the model then looks like this
```python
import torch
from transformers import BioGptConfig
from histogpt.models import HistoGPTForCausalLM, PerceiverResamplerConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

histogpt = HistoGPTForCausalLM(BioGptConfig(), PerceiverResamplerConfig())
histogpt = histogpt.to(device)

text = torch.randint(0, 42384, (1, 256)).to(device)
image = torch.rand(1, 1024, 768).to(device)

print(histogpt(text, image).logits.size())
```

Autoregressive text generation can be started with
```python
from histogpt.helpers.inference import generate

output = generate(
    model=histogpt,
    prompt=torch.randint(0, 42384, (1, 2)),
    image=torch.rand(1, 2, 768),
    length=256,
    top_k=40,
    top_p=0.95,
    temp=0.7,
    device=device
)

print(output.size())
```

After downloading the model weight, we can generate pathology reports from image features. A step-by-step guide is provided in the notebook "tutorial-1.ipynb".
```python
import h5py
from transformers import BioGptTokenizer

PATH = '/content/histogpt-1b-6k-pruned.pth?download=true'
state_dict = torch.load(PATH, map_location=device)
histogpt.load_state_dict(state_dict, strict=True)

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

prompt = 'Final diagnosis:'
prompt = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

with h5py.File('/content/2023-03-06 23.51.44.h5?download=true', 'r') as f:
    features = f['feats'][:]
    features = torch.tensor(features).unsqueeze(0).to(device)

output = generate(
    model=histogpt,
    prompt=prompt,
    image=features,
    length=256,
    top_k=40,
    top_p=0.95,
    temp=0.7,
    device=device
)

decoded = tokenizer.decode(output[0, 1:])
print(decoded)
```

To obtain the feature vectors, we use our extraction algorithm. An end-to-end example is provided in the notebook "tutorial-2.ipynb".
``` python
from histogpt.helpers.patching import main, PatchingConfigs

configs = PatchingConfigs()
configs.slide_path = '/content/slide_folder'
configs.save_path = '/content/save_folder'
configs.model_path = '/content/ctranspath.pth?download=true'
configs.patch_size = 256
configs.white_thresh = [170, 185, 175]
configs.edge_threshold = 2
configs.resolution_in_mpp = 0.0
configs.downscaling_factor = 4.0
configs.batch_size = 16

main(configs)
```
