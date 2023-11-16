from torch.nn.parallel import DistributedDataParallel
from megatron.model import GPTModel
from megatron.model.transformer import ParallelTransformerLayer
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
)
import argparse


def freeze_transformer_layers(
    args: argparse.Namespace,
    model: DistributedDataParallel,
) -> None:
    # type check
    if not isinstance(model.module.module, GPTModel):
        raise TypeError("Provided model is not an instance of GPTModel")

    assert args.freeze_transformer_layers % 2 == 0
    transformer_layers: ParallelTransformerLayer = model.module.module.language_model.encoder  # type: ignore

    pipeline_parallel_size: int = get_pipeline_model_parallel_world_size()
    tensor_parallel_size: int = get_pipeline_model_parallel_world_size()
    assert args.tensor_model_parallel_size == tensor_parallel_size

    pp_rank: int = get_pipeline_model_parallel_rank()

    num_layers_per_pp: int = args.num_layers // pipeline_parallel_size
    assert num_layers_per_pp == len(transformer_layers.layers)

    num_active_layers: int = args.num_layers - args.freeze_transformer_layers

    for i, layer in enumerate(transformer_layers.layers):
        layer_index: int = i + num_layers_per_pp * pp_rank
        if num_active_layers // 2 <= layer_index < args.num_layers - num_active_layers:
            for param in layer.parameters():
                param.requires_grad = False


def unfreeze_all_parameters(model: DistributedDataParallel) -> None:
    for param in model.parameters():
        param.requires_grad = True
