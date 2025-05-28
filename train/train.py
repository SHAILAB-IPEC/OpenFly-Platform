import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union
import datetime
import draccus
import torch
import torch.distributed as dist
import yaml
import transformers
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import HfArgumentParser, set_seed
from transformers.modeling_outputs import CausalLMOutputWithPast
from model.config import OpenFlyConfig
from model.overwatch import initialize_overwatch
from model.load_model import load_vla
from datasets import get_vla_dataset_and_collator, make_dataset_from_rlds, make_interleaved_dataset, get_oxe_dataset_kwargs_and_weights


from model.strategy import TrainingStrategy
from model.metrics import VLAMetrics
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics



@dataclass
class DataArguments:
    
    # Directory Paths
    data_root_dir: Path = Path(                                 
        "tensorflow_dataset"
    )
    run_root_dir: Path = Path("runs")                                                              

    data_mix: str = "vln_mix"        

    shuffle_buffer_size: int = 256_000  
    resume_step: Optional[int] = None                           
    resume_epoch: Optional[int] = None                          

    # Run Arguments                    
    save_interval: int = 5000                                   
    image_aug: bool = False                                           

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")              
    wandb_project: str = "openvla"                              
    wandb_entity: str = "stanford-voltron"                      


@dataclass
class TrainingArguments:
    
    seed: int = 7                                            
    pretrained_checkpoint: Optional[Path] = None   
    hf_token = ""                 
    is_resume: bool = True   

    freeze_vision_backbone: bool = False                   
    freeze_llm_backbone: bool = False                
    unfreeze_last_llm_layer: bool = False
    
    epochs: int = 1000                      
    max_steps: Optional[int] = None         

    expected_world_size: int = 8            
    global_batch_size: int = 128            
    per_device_batch_size: int = 16                     
    learning_rate: float = 2e-5 
    weight_decay: float = 0.0  
    max_grad_norm: float = 1.0 
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0  

    train_strategy: str = "fsdp-full-shard"  
    enable_gradient_checkpointing: bool = True      
    enable_mixed_precision_training: bool = True    
    reduce_in_full_precision: bool = True           

    grid_size: int = 16
    history_frames: int = 2
    # fmt: on



def main(data_args=None, training_args=None):

    # Initialize Overwatch =>> Wraps `logging.Logger`
    overwatch = initialize_overwatch(__name__)
    
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()
    run_id = (
        f"n{training_args.expected_world_size // 8}+b{training_args.per_device_batch_size}+x{training_args.seed}"
    )


    worker_init_fn = None

    os.makedirs(run_dir := (data_args.run_root_dir / run_id), exist_ok=True)
    os.makedirs(data_args.run_root_dir / run_id / "checkpoints", exist_ok=True)

    model = load_vla(training_args.pretrained_checkpoint, hf_token=training_args.hf_token, load_for_training=True, grid_size=training_args.grid_size)

    for param in model.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    # Determine training "stage" based on frozen vs unfrozen parameters --> supports different fine-tuning schemes!
    if not training_args.freeze_vision_backbone and not training_args.freeze_llm_backbone:
        stage = "vla-full-train"  # Full fine-tuning
    elif training_args.freeze_vision_backbone and not training_args.freeze_llm_backbone:
        stage = "vla-train"  # Frozen vision encoder
    elif not training_args.freeze_vision_backbone and training_args.freeze_llm_backbone:
        assert training_args.unfreeze_last_llm_layer, "You should unfreeze at least the last layer of your LLM!"
        stage = "vla-sandwich-train"  # Fine-tuning vision encoder, projector, and LLM last layer
    elif training_args.freeze_vision_backbone and training_args.freeze_llm_backbone:
        assert training_args.unfreeze_last_llm_layer, "Need to unfreeze at least last LLM layer to train!"
        stage = "vla-last-layer-train"  # Fine-tuning LLM last layer only
    else:
        raise ValueError(
            "Weight freezing configuration not supported. VLA config has the following parameters: "
            f"freeze_vision_backbone: {training_args.freeze_vision_backbone}"
            f"freeze_llm_backbone: {training_args.freeze_llm_backbone}"
            f"unfreeze_last_llm_layer: {training_args.unfreeze_last_llm_layer}"
        )

    # [Explicit] Call to `freeze_backbones` here for clarity =>> will log exactly what is/is not frozen
    overwatch.info(f"Stage Info: ")
    model.freeze_backbones(stage)

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )

    vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        data_args.data_root_dir,
        data_args.data_mix,
        image_transform=model.vision_backbone.get_image_transform(),
        tokenizer=model.llm_backbone.get_tokenizer(),
        default_image_resolution=model.vision_backbone.default_image_resolution,
        shuffle_buffer_size=data_args.shuffle_buffer_size,
        image_aug=data_args.image_aug,
    )

    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    train_strategy = TrainingStrategy(
        vlm=model,
        device_id=device_id,
        stage=stage,
        epochs=training_args.epochs,
        max_steps=training_args.max_steps,
        global_batch_size=training_args.global_batch_size,
        per_device_batch_size=training_args.per_device_batch_size,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        max_grad_norm=training_args.max_grad_norm,
        lr_scheduler_type=training_args.lr_scheduler_type,
        warmup_ratio=training_args.warmup_ratio,
        enable_gradient_checkpointing=training_args.enable_gradient_checkpointing,
        enable_mixed_precision_training=training_args.enable_mixed_precision_training,
        reduce_in_full_precision=training_args.reduce_in_full_precision,
        mixed_precision_dtype=training_args.mixed_precision_dtype,
        worker_init_fn=worker_init_fn,
        sharding_strategy="full-shard",
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(vla_dataset))

    metrics = VLAMetrics(
        data_args.trackers,
        run_id,
        run_dir,
        draccus.encode(data_args),
        wandb_project=data_args.wandb_project,
        wandb_entity=data_args.wandb_entity,
        resume_step=data_args.resume_step,
        resume_epoch=data_args.resume_epoch,
    )

    train_strategy.run_vla_training(
        vla_dataset,
        collator,
        action_tokenizer,
        metrics,
        save_interval=data_args.save_interval,
        history_frames=training_args.history_frames,
        grid_size=training_args.grid_size,
    )

    metrics.finalize()
    dist.barrier()
    dist.destroy_process_group()
    

if __name__ == '__main__':
    parser = transformers.HfArgumentParser(
        (DataArguments, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()
    main(data_args, training_args)
