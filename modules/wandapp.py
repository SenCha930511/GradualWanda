import os   
import numpy as np
import torch
import torch.nn.utils.prune as prune
from transformers import AutoTokenizer
from tqdm import tqdm
from lib.model_utils import get_llm, find_layers, check_sparsity
from lib.calibration_core import prepare_calibration_input
from lib.layerwrapper import WrappedGPT
from lib.data import get_loaders
from config import WandappConfig
import gc


def compute_gradient_metric(model, inputs):
    model.zero_grad()
    outputs = model(**inputs, labels=inputs["input_ids"], return_dict=True)
    loss = outputs.loss
    loss.backward()
    grad_metric = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_metric[name] = torch.abs(param.grad * param.data).detach()
    return grad_metric


def finalize_pruned_model(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                continue


def prune_wandapp(config: WandappConfig, model_path: str):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    print(f"Loading model {config.model}")
    model = get_llm(config.model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=False)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model.to(device)

    # calibration
    train_loader, _ = get_loaders("c4", nsamples=config.nsamples,
                                  seed=config.seed, seqlen=model.seqlen,
                                  tokenizer=tokenizer)
    with torch.no_grad():
        inps, outs, attn_mask, pos_ids = prepare_calibration_input(
            model, train_loader, device, batch_size=config.nsamples)
    if attn_mask is None:
        attn_mask = torch.ones((inps.size(0), model.seqlen), device=device)
    if pos_ids is None:
        pos_ids = torch.arange(model.seqlen, device=device).unsqueeze(0).repeat(inps.size(0), 1)

    # forward metric
    forward_metric = {}
    for i, layer in enumerate(tqdm(model.model.layers, desc="Forward metric")):
        layer.to(device)  # 把該層搬上 GPU
        subset = find_layers(layer)
        wrapped = {n: WrappedGPT(subset[n]) for n in subset}
        handles = [subset[n].register_forward_hook(
            lambda m, inp, out, name=n: wrapped[name].add_batch(inp[0].data, out.data))
                   for n in subset]
        for j in range(config.nsamples):
            _ = layer(inps[j].unsqueeze(0), attention_mask=attn_mask, position_ids=pos_ids)
        for h in handles: h.remove()
        for name in subset:
            scaler = wrapped[name].scaler_row.reshape(1, -1)
            act_var = wrapped[name].out_var.reshape(1, -1)
            W = torch.abs(subset[name].weight.data)
            forward_metric[f"{i}.{name}"] = (W * torch.sqrt(scaler) / (act_var + 1e-6))
        del wrapped
        gc.collect()
        torch.cuda.empty_cache()

    # gradient metric once
    print("Computing gradient metric...")
    inputs = {"input_ids": inps[:1].to(device),
              "attention_mask": attn_mask[:1].to(device),
              "position_ids": pos_ids[:1].to(device)}
    grad_metric = compute_gradient_metric(model, inputs)

    # pruning
    print("Pruning...")
    for i, layer in enumerate(model.model.layers):
        subset = find_layers(layer)
        for name, module in subset.items():
            key = f"{i}.{name}"
            W = module.weight.data
            F = forward_metric.get(key, torch.zeros_like(W))
            G = grad_metric.get(name, torch.zeros_like(W))
            score = F if not config.use_variant else 0.5 * F + 0.5 * G
            k = int(W.numel() * config.sparsity_ratio)
            thresh = torch.topk(score.view(-1), k, largest=False).values.max()
            mask = (score <= thresh).view_as(W)
            W[mask] = 0.0
        gc.collect()
        torch.cuda.empty_cache()

    overall_sparsity, _ = check_sparsity(model)
    print(f"[Wanda++] Overall sparsity: {overall_sparsity:.4f}")

    if config.save:
        os.makedirs(config.save, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config.save, "pytorch_model_pruned.bin"))
    if config.save_model:
        finalize_pruned_model(model)
        model.save_pretrained(config.save_model)
        tokenizer.save_pretrained(config.save_model)

    return overall_sparsity
