# HistoGPT

[[preprint]()] [[weights](https://huggingface.co/marr-peng-lab/histogpt)]  [[notebook](https://github.com/marrlab/HistoGPT/blob/main/histogpt_notebook.ipynb)]

## Generating highly accurate histopathology reports from whole slide images

HistoGPT is a vision language foundation model for dermatopathology. The model takes multiple tissue sections from a patient as input and generates a highly accurate pathology report that includes the disease classification, tumor subtype prediction, tumor thickness estimation, and other important clinical information. Most importantly, HistoGPT is fully interpretable, as every word or phrase in the output text can be visualized in the original image.

<img src="github/figure-1.png" width="800"/>

We trained HistoGPT on a large-scale dataset of 6,000 patient-report pairs from over 12,000 whole slide images (WSIs) of over 150 different skin conditions (healthy, inflammatory, cancerous, ...) provided by the Department of Dermatology at the Technical University of Munich (TUM). To test our model, we extensively evaluated HistoGPT on five external cohorts from five different countries, including a dataset of 1,300 patient-report pairs from the Department of Dermatology at the University Hospital Münster (UKM).

## HistoGPT learns from vision and language
HistoGPT takes an whole slide image as input, tiles it into smaller image patches, extracts the feature vectors for each image patch with an image encoder, reduces the number of feature vectors to a fixed size with a Perceiver Resampler, and combines them with text features coming from a language model via interleaved cross-attention layers.

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

out = generate(
    model=histogpt,
    prompt=torch.randint(0, 42384, (1, 2)),
    image=torch.rand(1, 2, 768),
    length=256,
    top_k=40,
    top_p=0.95,
    temp=0.7,
    device=device
)

print(out.size())
```

After downloading the model weight, we can generate pathology reports from image features. A step-by-step guide is provided in the notebook "histogpt_notebook.ipynb".
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

To obtain the feature vectors, we use our extraction algorithm.
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

## ToDo
- [x] make repository ready for publication
- [ ] implement ensemble refinement
- [ ] create an accessible zero-shot tool
- [ ] add visualization examples
