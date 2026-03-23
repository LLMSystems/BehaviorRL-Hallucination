from dataclasses import dataclass, field
from typing import List, Optional
import torch
import re 


@dataclass
class GRPOPathConfig:
    dataset_path: str = "/app/hallucination_evals/dataset/bonnieQA/dataset_for_grpo/bonnieQA_hallucination_training_dataset_rl_dataset_20260114_134126"
    # evaldataset_path: str = "/data/eddie.hsiao_data/Unsloth_tuner/data/hallucination_dpo/dpo_dataset_val.jsonl"
    output_dir: str = "outputs"

@dataclass
class GRPOModelConfig:
    # model_name: str = "/data/data_science_department/sharefolder/eddie/llm_ft/20260106-174521_halluciation_gemma3_12B_merg300"
    model_name: str = "/models/eddie/llm_ft/donttouch/halluciation_dpo_model"
    # model_name: str = "/data/data_science_department/sharefolder/max/llm/gemma-3-1b-it"
    max_seq_length: int = 4000
    dtype: str = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    full_finetuning: bool = False


@dataclass
class GRPOLoRAConfig:
    r: int = 32
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    lora_alpha: int = 64
    lora_dropout: float = 0
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[dict] = None

@dataclass
class GRPOTrainerConfig:
    per_device_train_batch_size: int = 1  
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    max_steps: int = 4000
    warmup_steps: int = 210  
    warmup_ratio: float = 0.1
    learning_rate: float = 5e-6  
    logging_steps: int = 1
    optim: str = "adamw_torch_fused"
    weight_decay: float = 0.1  
    lr_scheduler_type: str = "cosine"  
    seed: int = 42
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    # output_dir: str = "outputs"
    report_to: str = "tensorboard"
    # GRPO 特定參數
    num_generations: int = 8
    max_prompt_length: int = 3000
    max_completion_length: int = 256  # max_seq_length - max_prompt_length
    
    bf16: bool = field(default_factory=torch.cuda.is_bf16_supported)
    eval_strategy: str = "no"  
    eval_steps: int = 1
    save_strategy: str = "steps"
    save_steps: int = 100  # 每 1 步保存一次
    load_best_model_at_end: bool = False  # 沒有 eval 就不需要 best model
    max_grad_norm: float = 0.1  # 降低以更穩定
    gradient_checkpointing: bool = False  # 避免 cache warning 
    # importance_sampling_level: str = "sequence" 
    # vllm settings

@dataclass
class CallbackConfig:
    """
    回呼函式相關設定
    """
    early_stopping_patience: int = 20
    early_stopping_threshold: float = 0.0

'''
reward function for exact format matching
'''
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

def reward_function_1(completions, **kwargs):
    match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL
    )
    
    scores = []
    for i, completion in enumerate(completions):
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None:
            score += 3.0
        
        if 'avg_entropy' in kwargs:
            avg_entropy_list = kwargs.get('avg_entropy', [])
            if i < len(avg_entropy_list):
                avg_entropy = avg_entropy_list[i]
                score += avg_entropy
                print(f" Sample {i}: avg_entropy={avg_entropy:.4f}")
        
        scores.append(score)
    return scores