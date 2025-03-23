# lib/model_utils.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.seqlen = model.config.max_position_embeddings
    return model


def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    count, total_params = 0, 0

    layer_sparsity = []

    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        sub_count, sub_params = 0, 0
        for name in subset:
            W = subset[name].weight.data
            if W.device.type != 'meta':
                count += (W == 0).sum().item()
                total_params += W.numel()
                sub_count += (W == 0).sum().item()
                sub_params += W.numel()
        if sub_params > 0:
            layer_sparsity.append((i, float(sub_count)/sub_params))

    model.config.use_cache = use_cache
    overall_sparsity = float(count)/total_params if total_params > 0 else 0.0
    return overall_sparsity, layer_sparsity