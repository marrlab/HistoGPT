import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import top_k_top_p_filtering


def generate(model, device, prompt, image, length=256, top_k=40, top_p=0.95, temp=0.7):
    """  
    autoregressive generation using top-k and top-p sampling
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

            #logits[:, mask] = float('-inf')
            logits = top_k_top_p_filtering(logits=logits, top_k=40, top_p=0.95)

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
