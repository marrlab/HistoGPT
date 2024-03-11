""" 
HistoGPT Training Helper Functions
Author: Manuel Tran / Helmholtz Munich
"""

import torch


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print('Successfully saved!')


def load_model(model, dict_path):
    state_dict = torch.load(dict_path)
    model.load_state_dict(state_dict)
    print('Model loaded successfully!')
    return model


param_pairs = [
    ('self_attn.k_proj.weight', 'self_attn.k_proj.bias'),
    ('self_attn.v_proj.weight', 'self_attn.v_proj.bias'),
    ('self_attn.q_proj.weight', 'self_attn.q_proj.bias'),
    ('self_attn.out_proj.weight', 'self_attn.out_proj.bias'),
    ('self_attn_layer_norm.weight', 'self_attn_layer_norm.bias'),
    ('fc1.weight', 'fc1.bias'), ('fc2.weight', 'fc2.bias'),
    ('final_layer_norm.weight', 'final_layer_norm.bias')
]


def load_layer(layer, state_dict, layer_idx, prefix='biogpt.layers.'):
    layer_state_dict = layer.state_dict()
    for weight_name, bias_name in param_pairs:
        weight_key = f'{prefix}{layer_idx}.{weight_name}'
        bias_key = f'{prefix}{layer_idx}.{bias_name}'

        layer_state_dict[weight_name] = state_dict[weight_key]
        layer_state_dict[bias_name] = state_dict[bias_key]

    layer.load_state_dict(layer_state_dict)
    return layer


def load_sublayer(module, state_dict, keys):
    submodule_state_dict = {key.split('.')[-1]: state_dict[key] for key in keys}
    module.load_state_dict(submodule_state_dict)
