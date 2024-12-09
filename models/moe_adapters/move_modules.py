import logging
from typing import Literal

import einops
import torch
import torch.nn as nn
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrConfig, DeformableDetrDecoderLayer)


class ADT(nn.Module):
    def __init__(self, dim=1408, num_layers=6, origin_hw=16, num_queries=64):
        super().__init__()
        
        coarse_hw = int(num_queries**0.5)
        self.origin_hw = origin_hw
        self.num_queries = num_queries
        assert coarse_hw**2 == num_queries, f"num_queries({num_queries}) must be a square number"
        
        layer_config = DeformableDetrConfig(d_model=dim, num_queries=num_queries, num_feature_levels=1)
        self.layers = nn.ModuleList(
            [DeformableDetrDecoderLayer(config=layer_config) for _ in range(num_layers)]
        )
        
        self.reference_points = nn.Parameter(self._get_reference_points())
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_queries, dim))
        self.level_start_index = torch.tensor([0], dtype=torch.long)
        logging.info("Performing zero init on DSelector")
        self.apply(self._zero_init)
    
    def _zero_init(self, module):
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, DeformableDetrDecoderLayer):
            for l in [
                module.self_attn.out_proj,
                module.encoder_attn.output_proj,
                module.fc2
            ]:
                l.weight.data.zero_()
                l.bias.data.zero_()
    
    def _get_reference_points(self):
        steps = int(self.num_queries**0.5)
        grid = torch.meshgrid(
            torch.linspace(0, 1, steps), 
            torch.linspace(0, 1, steps), 
            indexing='ij'
        )
        reference_points = torch.stack(grid, dim=-1)  # [steps, steps, 2]
        reference_points = einops.rearrange(reference_points, 'h w c -> (h w) c')
        return reference_points.unsqueeze(0) # [1, num_queries, 2]
        
    def forward(self, hidden_states, inputs_embeds, spatial_shapes):
        """
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
        """
        # deformable attention only supports fp32
        batch_size = inputs_embeds.shape[0]
        # self.spatial_shapes=self.spatial_shapes.to(hidden_states.device)
        self.level_start_index=self.level_start_index.to(hidden_states.device)
        reference_points = self.reference_points.expand(batch_size, -1, -1)[:, :, None]
        for layer in self.layers:
            if spatial_shapes.shape[0] == 1 and spatial_shapes[0].prod().item()==inputs_embeds.shape[1]:
                hidden_states = layer(
                    hidden_states,
                    encoder_hidden_states=inputs_embeds,
                    position_embeddings=self.position_embeddings,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=self.level_start_index,
                )[0]
            else:
                assert spatial_shapes.shape[0] == batch_size
                hidden_states = torch.cat([
                    layer(
                        hidden_states[i:i+1],
                        encoder_hidden_states=inputs_embeds[i:i+1, :spatial_shapes[i].prod().item()],
                        position_embeddings=self.position_embeddings,
                        reference_points=reference_points[i:i+1],
                        spatial_shapes=spatial_shapes[i:i+1],
                        level_start_index=self.level_start_index,
                    )[0]
                    for i in range(batch_size)
                ], dim=0)
            
        return hidden_states


class MoEAggregator(nn.Module):
    """
    MoEAggregator is a module that aggregates the outputs of the experts
    
    num_experts (int): Number of experts
    seq_dim (int): The dimension of the input sequence
    router_method (str): The method to route the input sequence to the experts
    """
    def __init__(
        self, 
        num_experts=2, 
        seq_dim=256, 
        router_method: Literal["mlp", "add"]="mlp"
    ):
        super().__init__()
        self.num_experts = num_experts
        self.router_method = router_method
        self.routing_outcome = None
        if router_method == 'mlp':
            self.router = nn.Sequential(
                nn.Linear(seq_dim, seq_dim),
                nn.GELU(),
                nn.Linear(seq_dim, num_experts),
            )
        else:
            assert router_method == "add", f"router_method({router_method}) not supported"

    def forward(self, inputs_embeds, seq_embeds):
        """
        inputs_embeds (List of `torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`)
        """
        hidden_states = torch.zeros_like(inputs_embeds[0])
        bs = inputs_embeds[0].shape[0]
        self.routing_outcome = []
        if self.router_method == 'add':
            ratios = torch.ones(bs, len(inputs_embeds), device=inputs_embeds[0].device)
        else:
            ratios = self.router(seq_embeds)
            ratios = torch.softmax(ratios, dim=-1)
        for i in range(self.num_experts):
            _weighted_embeds = ratios[:,i].view(-1,1,1) * inputs_embeds[i]
            hidden_states += _weighted_embeds
            self.routing_outcome.append(_weighted_embeds.abs().mean(dim=-1).mean(dim=-1))
        self.routing_outcome = torch.stack(self.routing_outcome, dim=1)
        return hidden_states
