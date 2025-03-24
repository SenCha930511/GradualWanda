# api/wanda_api.py
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


def prune_wanda(config: PruningConfig):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

    print(f"Torch Version: {torch.__version__}")
    print(f"# of GPUs: {torch.cuda.device_count()}")

    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)

    prune_n, prune_m = 0, 0
    if config.sparsity_type != "unstructured":
        assert config.sparsity_ratio == 0.5, "N:M structured sparsity must be 0.5"
        prune_n, prune_m = map(int, config.sparsity_type.split(":"))

    print(f"Loading model {config.model}")
    model = get_llm(config.model, config.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in config.model or "65b" in config.model:
        device = model.hf_device_map["lm_head"]

    print(f"Using device: {device}")

    if config.sparsity_ratio != 0:
        print("Pruning starts")
        dataloader, _ = get_loaders("c4", nsamples=config.nsamples, seed=config.seed, seqlen=model.seqlen, tokenizer=tokenizer)
        print("Calibration data loaded")

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

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(config.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            for h in handles:
                h.remove()

            # 剪枝邏輯略 (可移到 lib/pruning_core 寫細部)

            inps, outs = outs, inps

            gc.collect()
            torch.cuda.empty_cache()

    overall_sparsity, layer_sparsity = check_sparsity(model)
    print("*" * 30)
    print(f"Overall sparsity: {overall_sparsity:.4f}")

    if config.save:
        if not os.path.exists(config.save):
            os.makedirs(config.save)
        save_filepath = os.path.join(config.save, f"log_wanda.txt")
        with open(save_filepath, "w") as f:
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            print(f"wanda\t{overall_sparsity:.4f}", file=f, flush=True)

    if config.save_model:
        model.save_pretrained(config.save_model)
        tokenizer.save_pretrained(config.save_model)