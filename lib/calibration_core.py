# lib/calibration_core.py
import torch
from torch import nn
from torch.cuda.amp import autocast


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
            autocast_context = autocast(dtype=torch.float16) if dtype == torch.float16 else torch.no_grad()
            with autocast_context:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device) if len(batch) > 1 else (input_ids != model.config.pad_token_id).long()
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