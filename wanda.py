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

# 設定 CUDA 的環境變數，調整記憶體分配策略
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# 輸出目前使用的套件版本與 GPU 數量，方便除錯與確認環境
print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


# ----------------------------
# 取得模型函數
# ----------------------------
def get_llm(model_name, cache_dir="llm_weights"):
    """
    從 HuggingFace 載入因果語言模型（Causal LM），並設定相關參數。
    
    參數:
      model_name: 模型名稱或路徑
      cache_dir: 模型下載或緩存的路徑
    
    傳回:
      載入好的模型，並將模型的最大序列長度設定為模型的最大位置編碼數量。
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,    # 使用 float16 以降低記憶體使用量
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True,         # 低 CPU 記憶體使用模式
        device_map="auto"               # 自動配置模型到可用裝置
    )

    # 將模型序列長度屬性設定為最大位置編碼數量
    model.seqlen = model.config.max_position_embeddings 
    return model


# ----------------------------
# 遞迴尋找指定類型層的函數
# ----------------------------
def find_layers(module, layers=[nn.Linear], name=''):
    """
    遞迴尋找 module 中所有指定類型（預設為 nn.Linear）的子層。
    
    參數:
      module: 要搜尋的模型層或模組
      layers: 要尋找的層類型清單（預設只尋找線性層）
      name: 用來記錄層的名稱（用於遞迴呼叫）
    
    傳回:
      一個字典，key 為層的名稱，value 為對應的層模組
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    # 遍歷所有子層，並遞迴搜尋
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


# ----------------------------
# 檢查模型剪枝後的稀疏度
# ----------------------------
def check_sparsity(model):
    """
    計算模型中所有線性層的零值比例（稀疏度），並依層級印出結果。
    
    參數:
      model: 已剪枝的模型
    
    傳回:
      整體稀疏度（零值數量 / 總參數數量）
    """
    # 暫時關閉 use_cache 以便取得所有參數
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    # 逐層檢查稀疏度
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        # 遍歷該層所有線性子層
        for name in subset:
            W = subset[name].weight.data
            # 檢查權重是否在實際裝置上（非 meta）
            if W.device.type != 'meta':
                count += (W == 0).sum().item()       # 累加零值數量
                total_params += W.numel()             # 累加總參數數量

                sub_count += (W == 0).sum().item()      # 當前子層零值數量
                sub_params += W.numel()                 # 當前子層參數總數
            else:
                print(f"Layer {i} {name} is on meta device, skipping sparsity check.")

        # 印出當前層的稀疏度
        if sub_params > 0:
            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")
        else:
            print(f"layer {i} sparsity check skipped.")

    # 還原 use_cache 設定
    model.config.use_cache = use_cache 
    if total_params > 0:
        return float(count)/total_params 
    else:
        return 0.0


# ----------------------------
# 準備校準資料的輸入
# ----------------------------
def prepare_calibration_input(model, dataloader, device, batch_size=1):
    """
    利用校準資料取得模型中間層的輸入資訊，供剪枝時調校參數使用。
    
    參數:
      model: 已載入的模型
      dataloader: 提供校準資料的 dataloader
      device: 使用的裝置（例如 GPU）
      batch_size: 批次大小（預設 1）
    
    傳回:
      inps: 收集到的中間層輸入張量
      outs: 與 inps 同型態的輸出張量（用作佔位）
      attention_mask: 注意力遮罩
      position_ids: 位置編碼張量
    """
    # 暫存原本的 use_cache 設定，並關閉以確保前向傳播時輸出正確
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # 如果模型 embed_tokens 層有指定裝置，則使用該裝置
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    # 取得模型參數資料型別
    dtype = next(iter(model.parameters())).dtype
    torch.cuda.empty_cache()
    
    # 建立儲存輸入資料的 tensor，形狀為 (batch_size, 序列長度, 隱藏層維度)
    inps = torch.zeros((batch_size, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    # 用來存放中間計算資訊的快取字典
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    # 定義一個 Catcher 模組，用來攔截第一層的輸入資料
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module  # 原本的層
        def forward(self, inp, attention_mask=None, position_ids=None, **kwargs):
            # 將攔截到的輸入存入 inps，並記錄 attention_mask 與 position_ids
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = attention_mask
            cache['position_ids'] = position_ids
            # 丟出異常以終止前向傳播
            raise ValueError

    # 將模型第一層替換成 Catcher，用以捕捉輸入資料
    layers[0] = Catcher(layers[0])
    try:
        # 遍歷校準資料批次
        for batch in dataloader:
            # 根據資料型別選擇適當的上下文（autocast 或 no_grad）
            if dtype == torch.float16:
                autocast_context = autocast(dtype=torch.float16)
            else:
                autocast_context = torch.no_grad()
            
            with autocast_context:
                # 將 batch 中的 input_ids 移至指定裝置
                input_ids = batch[0].to(device)
                
                # 如果 batch 包含 attention_mask，則轉移到裝置；否則自動產生
                if len(batch) > 1:
                    attention_mask = batch[1].to(device)
                else:
                    pad_token_id = model.config.pad_token_id if hasattr(model.config, 'pad_token_id') else 0
                    attention_mask = (input_ids != pad_token_id).long()
                
                # 產生位置編碼張量
                position_ids = torch.arange(model.seqlen, device=device).unsqueeze(0).repeat(input_ids.size(0), 1)
                
                # 前向傳播，預期會被 Catcher 攔截後拋出 ValueError
                model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
    except ValueError:
        # 捕捉到 ValueError 表示已成功取得所需的中間輸入，故中斷迴圈
        pass 
    # 還原第一層為原始模組
    layers[0] = layers[0].module

    # 建立與 inps 相同形狀的 tensor 作為佔位的輸出
    outs = torch.zeros_like(inps)
    # 從 cache 取得 attention_mask 與 position_ids
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    # 還原原本的 use_cache 設定
    model.config.use_cache = use_cache

    torch.cuda.empty_cache()

    return inps, outs, attention_mask, position_ids 


# ----------------------------
# 根據 alpha 值計算剪枝 mask 與當前稀疏度
# ----------------------------
def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    """
    依據給定的 alpha 值計算剪枝的閥值，並回傳剪枝 mask 與當前稀疏度。
    
    參數:
      alpha: 控制剪枝比例的係數
      sort_res: 經過排序後的權重資訊（包含排序值與索引）
      W_metric: 權重重要性評分
      tmp_metric: 權重累積和
      sum_before: 每一列權重評分總和
    
    傳回:
      W_mask: 剪枝 mask，True 表示該位置要剪除（設為 0）
      cur_sparsity: 目前剪枝後的稀疏度比例
    """
    # 計算每一列的累積門檻值
    thres_cumsum = sum_before * alpha 
    # 根據累積和與門檻值，計算出剪枝位置的布林遮罩
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    # 依據遮罩取得門檻值
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdim=True)-1)
    # 剪枝 mask 為權重重要性評分低於門檻值的位置
    W_mask = (W_metric <= thres)
    # 計算當前稀疏度比例
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


# ----------------------------
# 主剪枝函數 prune_wanda
# ----------------------------
def prune_wanda(config: PruningConfig):
    """
    根據給定的 PruningConfig 進行模型剪枝，包括校準資料的取得、剪枝 mask 的計算與模型儲存。
    
    參數:
      config: 剪枝設定，包含模型名稱、剪枝參數、校準樣本數、儲存路徑等資訊
    """
    # 設定隨機種子以利結果重現
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)

    # 處理 n:m sparsity 的情況，若不是 unstructured 模式則拆解 n:m
    prune_n, prune_m = 0, 0
    if config.sparsity_type != "unstructured":
        # 對於結構化 N:M 稀疏，需要稀疏比例為 0.5
        assert config.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, config.sparsity_type.split(":"))

    # 載入模型與 tokenizer
    print(f"loading llm model {config.model}")
    model = get_llm(config.model, config.cache_dir)
    model.eval()  # 設定為 evaluation 模式
    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=False)

    # 設定運算裝置，預設使用 cuda:0；若為 30b 或 65b 模型則根據 device_map 使用多 GPU
    device = torch.device("cuda:0")
    if "30b" in config.model or "65b" in config.model:
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    # 當稀疏比例不為 0 時，執行剪枝流程
    if config.sparsity_ratio != 0:
        print("pruning starts")
        # 暫存原本的 use_cache 設定，並關閉以確保前向傳播的正確性
        use_cache = model.config.use_cache 
        model.config.use_cache = False 

        # 載入校準資料
        print("loading calibration data")
        dataloader, _ = get_loaders("c4", nsamples=config.nsamples, seed=config.seed, seqlen=model.seqlen, tokenizer=tokenizer)
        print("dataset loading complete")
        # 透過校準資料取得中間層輸入與輸出資訊
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, batch_size=config.nsamples)

        # 若 attention_mask 或 position_ids 尚未產生，則自動建立
        if attention_mask is None:
            attention_mask = torch.ones((inps.size(0), model.seqlen), dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = torch.arange(model.seqlen, device=device).unsqueeze(0).repeat(inps.size(0), 1)

        # 取得模型所有 transformer 層
        layers = model.model.layers
        # 對每一層進行剪枝
        for i in tqdm.tqdm(range(len(layers))):
            layer = layers[i]
            subset = find_layers(layer)  # 取得該層所有線性層

            # 若該層在不同 GPU 上，則將校準資料移至對應裝置
            if f"model.layers.{i}" in model.hf_device_map:
                dev = model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

            # 用 WrappedGPT 包裹所有線性層，方便後續收集統計資訊
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            # 定義 forward hook，用以收集每個線性層的輸入與輸出
            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            # 為所有子層註冊 forward hook
            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            # 利用校準資料進行前向傳播，並藉由 hook 收集資訊
            for j in range(config.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            # 移除所有 hook
            for h in handles:
                h.remove()

            # 依據收集的資訊對每個子層進行剪枝
            for name in subset:
                # 計算權重的重要性評分，這裡用權重絕對值乘上對應 row 的 scaler（開根號）
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                # 初始化剪枝 mask，預設全部為 False
                W_mask = (torch.zeros_like(W_metric) == 1)
                # 若有 n:m 結構化稀疏設定，則逐列進行剪枝
                if prune_n != 0:
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii:(ii + prune_m)].float()
                            # 依據 topk 找出要保留的索引位置，將其他位置設為剪枝 mask True
                            W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
                else:
                    # 若非結構化稀疏，則進行全局排序後依據 variant 計算剪枝 mask
                    try:
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    except RuntimeError as e:
                        print(f"Sort failed for layer {i} name {name} with error: {e}")
                        continue

                    if config.use_variant:
                        # variant 方法：利用累積和與 alpha 調整剪枝比例
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)
                        alpha = 0.4
                        alpha_hist = [0., 0.8]
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        # 利用二分法調整 alpha 直到達到目標稀疏度
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
                        # 若不使用 variant，直接依據目標稀疏比例選取排序前的索引
                        indices = sort_res[1][:, :int(W_metric.shape[1] * config.sparsity_ratio)]
                        W_mask.scatter_(1, indices, True)

                # 將剪枝 mask 為 True 的權重位置設為 0
                subset[name].weight.data[W_mask] = 0

                # 釋放暫存變數，並清理 CUDA 記憶體
                del W_metric, sort_res
                gc.collect()
                torch.cuda.empty_cache()

            # 釋放包裹層與 hook 的資源
            del wrapped_layers
            del handles
            gc.collect()
            torch.cuda.empty_cache()

            # 將當前層的輸出作為下一層的輸入（交換 inps 與 outs）
            inps, outs = outs, inps

        # 還原 use_cache 設定
        model.config.use_cache = use_cache 
        torch.cuda.empty_cache()

    # 印出剪枝後模型的整體稀疏度
    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")

    # 若設定儲存路徑，則將剪枝結果寫入 log 檔案
    if config.save:
        if not os.path.exists(config.save):
            os.makedirs(config.save)
        save_filepath = os.path.join(config.save, f"log_{config.prune_method}.txt")
        with open(save_filepath, "w") as f:
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            print(f"{config.prune_method}\t{sparsity_ratio:.4f}", file=f, flush=True)

    # 若設定儲存模型，則將剪枝後的模型與 tokenizer 儲存
    if config.save_model:
        model.save_pretrained(config.save_model)
        tokenizer.save_pretrained(config.save_model)
