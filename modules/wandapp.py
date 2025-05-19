# modules/wanda_pp.py
import os
import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from lib.model_utils import get_llm, find_layers, check_sparsity
from lib.calibration_core import prepare_calibration_input
from lib.layerwrapper import WrappedGPT
from lib.data import get_loaders
from config import PruningConfig
import gc

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdim=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def adaptive_alpha_search(target_sparsity, sort_res, W_metric, tmp_metric, sum_before, max_iter=10):
    alpha = 0.4
    for _ in range(max_iter):
        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
        error = cur_sparsity - target_sparsity
        if abs(error) < 0.001:
            break
        alpha -= error * 0.5
        alpha = max(0.0, min(1.0, alpha))
    return W_mask, cur_sparsity, alpha

def prune_wandapp(config: PruningConfig, model_path: str):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    prune_n, prune_m = 0, 0
    if config.sparsity_type != "unstructured":
        assert config.sparsity_ratio == 0.5, "N:M structured sparsity must be 0.5"
        prune_n, prune_m = map(int, config.sparsity_type.split(":"))

    print(f"Loading model {config.model}")
    model = get_llm(config.model, model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in config.model or "65b" in config.model:
        device = model.hf_device_map["lm_head"]

    if config.sparsity_ratio != 0:
        print("Pruning starts")
        dataloader, _ = get_loaders("c4", nsamples=config.nsamples, seed=config.seed, seqlen=model.seqlen, tokenizer=tokenizer)
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, batch_size=config.nsamples)

        if attention_mask is None:
            attention_mask = torch.ones((inps.size(0), model.seqlen), dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = torch.arange(model.seqlen, device=device).unsqueeze(0).repeat(inps.size(0), 1)

        layers = model.model.layers
        for i in tqdm(range(len(layers))):
            layer = layers[i]
            subset = find_layers(layer)

            if f"model.layers.{i}" in model.hf_device_map:
                dev = model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

            wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = [subset[name].register_forward_hook(add_batch(name)) for name in wrapped_layers]

            for j in range(config.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            for h in handles:
                h.remove()

            for name in subset:
                # 加入 activation variance 作為敏感度指標
                scaler_row = wrapped_layers[name].scaler_row.reshape(1, -1)
                activation_var = wrapped_layers[name].out_var.reshape(1, -1)
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(scaler_row) / (activation_var + 1e-6)

                W_mask = (torch.zeros_like(W_metric) == 1)
                if prune_n != 0:
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii:(ii + prune_m)].float()
                            W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
                else:
                    try:
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    except RuntimeError as e:
                        print(f"Sort failed for layer {i} name {name}: {e}")
                        continue

                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    W_mask, cur_sparsity, final_alpha = adaptive_alpha_search(config.sparsity_ratio, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"[Layer {i} - {name}] alpha: {final_alpha:.4f}  sparsity: {cur_sparsity:.4f}")

                subset[name].weight.data[W_mask] = 0

                del W_metric, sort_res
                gc.collect()
                torch.cuda.empty_cache()

            del wrapped_layers
            del handles
            inps, outs = outs, inps
            gc.collect()
            torch.cuda.empty_cache()

    overall_sparsity, layer_sparsity = check_sparsity(model)
    print("=" * 30)
    print(f"[Wanda++] Overall sparsity: {overall_sparsity:.4f}")

    if config.save:
        os.makedirs(config.save, exist_ok=True)
        log_path = os.path.join(config.save, f"log_wandapp.txt")
        with open(log_path, "w") as f:
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            print(f"wandapp\t{overall_sparsity:.4f}", file=f, flush=True)

    if config.save_model:
        model.save_pretrained(config.save_model)
        tokenizer.save_pretrained(config.save_model)

    return overall_sparsity
