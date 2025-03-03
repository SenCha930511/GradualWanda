import os 
import numpy as np
import torch
import gc
import tqdm
import torch.nn as nn 
from lib.sparsegpt import SparseGPT
from lib.data import get_loaders
from lib.layerwrapper import WrappedGPT
from lib.pruning_config import PruningConfig
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# 基礎資訊
print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


# 取得模型
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
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            if W.device.type != 'meta':
                count += (W == 0).sum().item()
                total_params += W.numel()

                sub_count += (W == 0).sum().item()
                sub_params += W.numel()
            else:
                print(f"Layer {i} {name} is on meta device, skipping sparsity check.")

        if sub_params > 0:
            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")
        else:
            print(f"layer {i} sparsity check skipped.")

    model.config.use_cache = use_cache 
    if total_params > 0:
        return float(count)/total_params 
    else:
        return 0.0


def prepare_calibration_input(model, dataloader, device, batch_size=1):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    torch.cuda.empty_cache()
    
    inps = torch.zeros((batch_size, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, attention_mask=None, position_ids=None, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = attention_mask
            cache['position_ids'] = position_ids
            raise ValueError

    layers[0] = Catcher(layers[0])
    try:
        for batch in dataloader:
            if dtype == torch.float16:
                autocast_context = autocast(dtype=torch.float16)
            else:
                autocast_context = torch.no_grad()
            
            with autocast_context:
                input_ids = batch[0].to(device)
                
                if len(batch) > 1:
                    attention_mask = batch[1].to(device)
                else:
                    pad_token_id = model.config.pad_token_id if hasattr(model.config, 'pad_token_id') else 0
                    attention_mask = (input_ids != pad_token_id).long()
                
                position_ids = torch.arange(model.seqlen, device=device).unsqueeze(0).repeat(input_ids.size(0), 1)
                
                model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
    except ValueError:
        pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    torch.cuda.empty_cache()

    return inps, outs, attention_mask, position_ids 


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdim=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_wanda(config: PruningConfig):
    # 設定隨機種子以利結果重現
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)

    # 處理 n:m sparsity
    prune_n, prune_m = 0, 0
    if config.sparsity_type != "unstructured":
        assert config.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, config.sparsity_type.split(":"))

    # 載入模型與 tokenizer
    print(f"loading llm model {config.model}")
    model = get_llm(config.model, config.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=False)

    # 設定運算裝置
    device = torch.device("cuda:0")
    if "30b" in config.model or "65b" in config.model:  # 對於 30b 與 65b 模型使用多 GPU
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if config.sparsity_ratio != 0:
        print("pruning starts")
        use_cache = model.config.use_cache 
        model.config.use_cache = False 

        # 載入校準資料
        print("loading calibration data")
        dataloader, _ = get_loaders("c4", nsamples=config.nsamples, seed=config.seed, seqlen=model.seqlen, tokenizer=tokenizer)
        print("dataset loading complete")
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, batch_size=config.nsamples)

        if attention_mask is None:
            attention_mask = torch.ones((inps.size(0), model.seqlen), dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = torch.arange(model.seqlen, device=device).unsqueeze(0).repeat(inps.size(0), 1)

        layers = model.model.layers
        for i in tqdm.tqdm(range(len(layers))):
            layer = layers[i]
            subset = find_layers(layer)

            # 如有必要，將資料轉移至指定裝置
            if f"model.layers.{i}" in model.hf_device_map:
                dev = model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

            # 用 WrappedGPT 包裹所有線性層
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

            # 前向傳播收集校準資料
            for j in range(config.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            # 對每個子層進行剪枝
            for name in subset:
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
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
                        print(f"Sort failed for layer {i} name {name} with error: {e}")
                        continue

                    if config.use_variant:
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)
                        alpha = 0.4
                        alpha_hist = [0., 0.8]
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        while (torch.abs(cur_sparsity - config.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > config.sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha
                            alpha = alpha_new 
                            W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                    else:
                        indices = sort_res[1][:, :int(W_metric.shape[1] * config.sparsity_ratio)]
                        W_mask.scatter_(1, indices, True)

                # 將 mask 為 True 的位置設為 0
                subset[name].weight.data[W_mask] = 0

                del W_metric, sort_res
                gc.collect()
                torch.cuda.empty_cache()

            del wrapped_layers
            del handles
            gc.collect()
            torch.cuda.empty_cache()

            # 交換 inps 與 outs 以便下一層使用
            inps, outs = outs, inps

        model.config.use_cache = use_cache 
        torch.cuda.empty_cache()

    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")

    # 儲存剪枝結果與模型（如果有設定）
    if config.save:
        if not os.path.exists(config.save):
            os.makedirs(config.save)
        save_filepath = os.path.join(config.save, f"log_{config.prune_method}.txt")
        with open(save_filepath, "w") as f:
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            print(f"{config.prune_method}\t{sparsity_ratio:.4f}", file=f, flush=True)

    if config.save_model:
        model.save_pretrained(config.save_model)
        tokenizer.save_pretrained(config.save_model)
