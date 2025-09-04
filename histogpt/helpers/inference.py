""" 
HistoGPT Inference Helper Functions
Author: Manuel Tran / Helmholtz Munich
"""

import h5py
import openai
import random
import time
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import PreTrainedTokenizer
from ..clam.wsi_core.WholeSlideImage import WholeSlideImage
from torch import Tensor

def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

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

            pred = torch.multinomial(probs, num_samples=1)
            #pred = torch.argmax(logits, dim=1)

            if pred == 2:  # break at end token '</s>'
                break

            #if pred == 4:  # break at period token '.'
            #    break

            #if pred == 518:  # break at millimeter token 'mm'
            #    break

            out = torch.cat((out, pred.unsqueeze(0)), dim=1)

    return out


def chat_gpt(prompt, temperature, top_p):
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        temperature=temperature,
        top_p=top_p,
        n=1,
    )
    return [response.choices[i].message.content for i in range(len(response.choices))]


def api_call(prompt, retries, temperature, top_p):
    for i in range(retries):
        try:
            response = chat_gpt(prompt, temperature, top_p)
            return response
        except Exception as E:
            wait_time = (2**i) + random.random()
            print(f"Error occurred: {E}. Retry #{i+1} in {wait_time} seconds.")
            time.sleep(wait_time)
    raise Exception("Max retries exceeded.")


def ensemble_refinement(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompt: torch.tensor,
    image: torch.tensor,
    length: int,
    top_k: int = 40,
    top_p: float = 0.95,
    temp: float = 1.0,
    device: str = 'cuda',
    num_samples: int = 10,
    instruction: str = None,
    gpt_temp: float = 1.0,
    gpt_top_p: float = 1.0,
    retries: int = 10,
):
    if instruction == None:
        instruction = ("Summarize the following text. Be as accurate as possible!")

    outputs = []
    for _ in tqdm(range(num_samples)):
        output = generate(
            model=model,
            prompt=prompt,
            image=image,
            length=length,
            top_k=top_k,
            top_p=top_p,
            temp=temp,
            device=device
        )
        outputs.append(output)
    outputs = torch.concat(outputs, 1)

    prompt = instruction + tokenizer.decode(outputs[0, 1:])
    return api_call(prompt, retries, gpt_temp, gpt_top_p)[0]


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
