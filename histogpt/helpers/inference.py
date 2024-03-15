""" 
HistoGPT Inference Helper Functions
Author: Manuel Tran / Helmholtz Munich
"""

import h5py
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import top_k_top_p_filtering
from ..clam.wsi_core.WholeSlideImage import WholeSlideImage


def generate(
    model, prompt, image, length=256, top_k=40, top_p=0.95, temp=0.7, device='cuda'
):
    """  
    autoregressive generation of reports using top-k, top-p, and temperature sampling
    """
    model.eval()
    image = image.to(device)
    out = prompt.to(device)

    with torch.no_grad():
        for _ in tqdm(range(length), leave=False):
            inputs = out

            if device == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(inputs, image.float()).logits
                    logits = logits[:, -1, :] / temp
            else:
                logits = model(inputs, image.float()).logits
                logits = logits[:, -1, :] / temp

            #logits[:, mask] = float('-inf')
            logits = top_k_top_p_filtering(logits=logits, top_k=top_k, top_p=top_p)

            probs = F.softmax(logits, dim=-1)
            probs = probs.squeeze(0)

            #pred = torch.multinomial(probs, num_samples=1)
            pred = torch.argmax(logits, dim=1)

            if pred == 2:  # break at end token '</s>'
                break

            #if pred == 4:  # break at period token '.'
            #    break

            #if pred == 518:  # break at millimeter token 'mm'
            #    break

            out = torch.cat((out, pred.unsqueeze(0)), dim=1)

    return out


def visualize(
    model, tokenizer, source, target, feats_path, slide_path, save_path, device='cuda'
):
    """
    visualize target words or phrases from the source report as features in the input
    """

    with h5py.File(feats_path, 'r') as f:
        coordinates = f['coords'][:]
        features = f['feats'][:]
    coordinates = coordinates[:, 1:]

    leaf_tensor = torch.tensor(features, requires_grad=True)
    input_tensor = leaf_tensor.unsqueeze(0).to(device)

    source = tokenizer.encode(source)
    target = tokenizer.encode(target)

    start = [
        i for i, token in enumerate(source)  # extract token positions
        if token == target[0] and source[i:i + len(target)] == target
    ]
    end = [i + len(target) - 1 for i in start]

    source = torch.tensor([source])
    token_positions = list(range(start[0], end[0] + 1))

    # perform forward pass
    _ = model(source.to(device), input_tensor.float().to(device)).logits

    attention = model.histogpt.layers[-1][0].attn.attn[0]
    attention = attention[:, token_positions, :].clamp(min=0).mean(dim=(0, 1))
    attention = (attention - attention.min()) / (attention.max() - attention.min())

    perceive = model.histogpt.perceiver_resampler(input_tensor)
    gradient = torch.zeros(640, features.shape[0])

    for i in tqdm(range(640)):
        model.histogpt.perceiver_resampler.zero_grad()
        specific_output = perceive[0, 0, i, 0]
        specific_output.backward(retain_graph=True)
        gradient[i] = leaf_tensor.grad.norm(dim=1)

    scores = gradient.abs().mean(dim=1)
    scores = scores / scores.sum()
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    scores = scores * (attention.cpu()**10)
    scores = (gradient.T * scores).sum(dim=1)

    scores = scores.cpu().detach().numpy() * 100
    coordinates = coordinates[:, [1, 0]]

    wsi_object = WholeSlideImage(slide_path)
    wsi = wsi_object.getOpenSlide()
    vis_level = wsi.get_best_level_for_downsample(64)
    best_level = wsi_object.wsi.get_best_level_for_downsample(64)
    wsi_object.segmentTissue(
        best_level, filter_params={
            'a_t': 100,
            'a_h': 16,
            'max_n_holes': 10
        }
    )
    viz = wsi_object.visHeatmap(
        scores=scores,
        coords=coordinates * 4,
        vis_level=vis_level,
        patch_size=(1024, 1024),
        overlap=0.0,
        alpha=0.6,
        segment=False,
        cmap='Spectral_r'
    )
    viz[1].save(save_path)
