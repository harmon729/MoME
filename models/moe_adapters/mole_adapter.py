import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.llama.modeling_llama import LlamaMLP


def top1_gate(logits):
    y_soft = logits.softmax(dim=-1)
    gate, index = y_soft.max(dim=-1, keepdim=True)
    lb_loss = gate.sum(dim=-1).mean()
    y_hard = torch.zeros_like(logits, device=gate.device).scatter_(-1, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret, lb_loss


class Adapter(nn.Module):
    def __init__(self,
                 d_model=None,
                 bottleneck=64,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="learnable_scalar",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

class AdapterRouterMLP(nn.Module):
    def __init__(self,
        d_model,
        bottleneck=64,
        dropout=0.0,
        init_option="lora",
        adapter_scalar="learnable_scalar",
        adapter_layernorm_option="none",
        num_adapters = 2,
        sparse=True,
    ):
        super().__init__()
        self.adapters = nn.ModuleList([])
        for _ in range(num_adapters):
            self.adapters.append(Adapter(
                d_model=d_model,
                bottleneck=bottleneck,
                dropout=dropout,
                init_option=init_option,
                adapter_scalar=adapter_scalar,
                adapter_layernorm_option=adapter_layernorm_option
            ))
        self.router_mlp = nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Linear(384, num_adapters)
        )
        self.num_adapters = num_adapters
        self.router_embedding = None
        self.sparse = sparse
        self.routing_outcome = None # For Visualization
        
    def forward(self, x, add_residual=True, residual=None):
        assert self.router_embedding is not None
        if self.sparse:
            ratios, self.lb_loss = top1_gate(self.router_mlp(self.router_embedding))
        else:
            ratios = self.router_mlp(self.router_embedding).softmax(dim=-1)
        self.routing_outcome = ratios
        # Support Beam Search
        assert x.shape[0] % self.router_embedding.shape[0] == 0, f"x.shape[0]({x.shape[0]}) must be divisible by batch size({self.router_embedding.shape[0]})"
        num_beams = x.shape[0] // self.router_embedding.shape[0]
        ratios = ratios.repeat_interleave(num_beams, dim=0)
        
        output = 0
        for i in range(self.num_adapters):
            output += self.adapters[i](x, add_residual, residual) * ratios[:,i].view(-1,1,1)
        return output
    

def forward_mlp_llama(self, x):
    adapt_x = self.adapter(x, add_residual=False)
    x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return x + adapt_x

def set_moe_mlp_adapter_llama(model: nn.Module, d_model:int, bottleneck:int=64, num_adapters:int=3):
    for c in model.children():
        if type(c) == LlamaMLP:
            c.adapter = AdapterRouterMLP(d_model=d_model, bottleneck=bottleneck, num_adapters=num_adapters)
            bound_method = forward_mlp_llama.__get__(c, c.__class__)
            setattr(c, 'forward', bound_method)
        elif len(list(c.children())) != 0:
            set_moe_mlp_adapter_llama(c, d_model, bottleneck, num_adapters)

def set_router_embedding_mlp(model: nn.Module, router_embedding: torch.Tensor):
    for c in model.children():
        if type(c) == AdapterRouterMLP:
            c.router_embedding = router_embedding
        elif len(list(c.children())) != 0:
            set_router_embedding_mlp(c, router_embedding)