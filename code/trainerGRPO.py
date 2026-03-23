import os
# 繞過 FIPS 模式限制（必須在導入其他模組前設置）
os.environ['OPENSSL_CONF'] = '/dev/null'
os.environ['OPENSSL_FIPS'] = '0'

# 禁用 torch.compile 以避免編譯錯誤
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from unsloth.chat_templates import get_chat_template

import torch
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.utils import entropy_from_logits
import time

from datasets import Dataset
import pandas as pd
import dataclasses
from transformers import EarlyStoppingCallback

from datetime import datetime
import logging
import psutil
from transformers import TrainerCallback 
import os
import gc

import torch
from datasets import load_dataset, load_from_disk
from unsloth import FastLanguageModel
import types

from .hallucination.reward_function_advance_v2 import hallucination_reward_function
from .hallucination.config.config_grpo import GRPOPathConfig, GRPOModelConfig, GRPOLoRAConfig, GRPOTrainerConfig, CallbackConfig

class CustomLoggingCallback(TrainerCallback):
    """自訂的回呼函式，用於在訓練過程中記錄 loss 和資源使用情況。"""
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        """在 trainer 記錄日誌時觸發 (頻率由 logging_steps 控制)。"""
        if state.is_world_process_zero and logs is not None:
            self.logger.log_resource_usage(step=state.global_step)
            if 'loss' in logs:
                lr = logs.get('learning_rate')
                lr_str = f"{lr:.2e}" if lr is not None else "N/A"
                self.logger.logger.info(f"Step {state.global_step} | Training Loss: {logs['loss']:.4f} | Learning Rate: {lr_str}")

            # 記錄評估指標
            if 'eval_loss' in logs:
                self.logger.logger.info(f"Step {state.global_step} | Evaluation Loss: {logs['eval_loss']:.4f}")

class ResourceLogger:
    def __init__(self, log_dir='output'):
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger("ResourceLogger")
        self.logger.setLevel(logging.INFO)
        # 防止重複添加 handler
        if not self.logger.handlers:
            log_file = os.path.join(log_dir, 'train_resource.log')
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_resource_usage(self, step=None):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                free = total - reserved
                self.logger.info(
                    f"Step {step} | GPU {i}: total={total:.2f}GB, reserved={reserved:.2f}GB, allocated={allocated:.2f}GB, free={free:.2f}GB"
                )
        mem = psutil.virtual_memory()
        self.logger.info(f"Step {step} | CPU 記憶體: {mem.percent}% 已用, {mem.used/1024**3:.2f}GB/{mem.total/1024**3:.2f}GB")

class UnslothTrainer:
    def __init__(self, output_dir="outputs"):
        # 建立主資料夾
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # 建立 config 子資料夾
        self.config_dir = os.path.join(output_dir, "config")
        os.makedirs(self.config_dir, exist_ok=True)

        # 設定模型與 tokenizer 儲存路徑
        self.model_dir = os.path.join(output_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)

        # 設定 log 檔案路徑
        self.resource_logger = ResourceLogger(log_dir=output_dir)

        # 建立設定物件
        self.path_cfg = GRPOPathConfig(
                    output_dir=output_dir
        )
        self.model_cfg = GRPOModelConfig()
        self.lora_cfg = GRPOLoRAConfig()
        self.train_args_cfg = GRPOTrainerConfig()
        self.callback_cfg = CallbackConfig()
        # 設定 gemma3 chat template 套用
        self.gemma3_template = False

        # 儲存 config 設定檔
        import json
        with open(os.path.join(self.config_dir, "path_config.json"), "w") as f:
            json.dump(dataclasses.asdict(self.path_cfg), f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.config_dir, "model_config.json"), "w") as f:
            json.dump(dataclasses.asdict(self.model_cfg), f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.config_dir, "lora_config.json"), "w") as f:
            json.dump(dataclasses.asdict(self.lora_cfg), f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.config_dir, "train_args_config.json"), "w") as f:
            json.dump(dataclasses.asdict(self.train_args_cfg), f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.config_dir, "callback_config.json"), "w") as f:
            json.dump(dataclasses.asdict(self.callback_cfg), f, ensure_ascii=False, indent=2)

        # 載入資料集
        dataset_path = self.path_cfg.dataset_path
        dataset = load_from_disk(dataset_path)
        # dataset = dataset.shuffle(seed=42)  # Shuffle 數據集
        # eval_dataset = dataset.train_test_split(test_size=0.01, seed=42)['test']  # 1% 作為 eval 集
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_cfg.model_name,
            max_seq_length=self.model_cfg.max_seq_length,
            load_in_4bit=self.model_cfg.load_in_4bit,
            fast_inference = True,
            gpu_memory_utilization = 0.85, # Reduce if out of memory
        )

        # 處理多模態 Processor 問題（gemma-3-4b-it 使用 Gemma3Processor）
        # 確保使用純 tokenizer 而非 processor
        if hasattr(self.tokenizer, 'tokenizer'):
            print(f"✓ 檢測到 Processor，提取純 tokenizer 用於 GRPO")
            self.actual_tokenizer = self.tokenizer.tokenizer
        else:
            self.actual_tokenizer = self.tokenizer
            print(f"✓ 使用標準 tokenizer")

        # Disable cache to avoid warning with gradient checkpointing
        self.model.config.use_cache = False
        if hasattr(self.model, 'generation_config'):
            self.model.generation_config.use_cache = False
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            target_modules = self.lora_cfg.target_modules,
            finetune_vision_layers = self.lora_cfg.finetune_vision_layers,  # Turn off for just text!
            finetune_language_layers = self.lora_cfg.finetune_language_layers,  # Should leave on!
            finetune_attention_modules = self.lora_cfg.finetune_attention_modules,  # Attention good for GRPO
            finetune_mlp_modules = self.lora_cfg.finetune_mlp_modules,  # Should leave on always!
            r = self.lora_cfg.r,  # Larger = higher accuracy, but might overfit
            lora_alpha = self.lora_cfg.lora_alpha,  # Recommended alpha == r at least
            lora_dropout = self.lora_cfg.lora_dropout,
            bias = self.lora_cfg.bias,
            random_state = self.lora_cfg.random_state,
            use_gradient_checkpointing = "unsloth"
        )
        self.model.config.use_cache = False
        if hasattr(self.model, 'generation_config'):
            self.model.generation_config.use_cache = False
        if hasattr(self.model, 'base_model'):
            self.model.base_model.config.use_cache = False


        # early_stopping_callback = EarlyStoppingCallback(
        #     early_stopping_patience=self.callback_cfg.early_stopping_patience,
        #     early_stopping_threshold=self.callback_cfg.early_stopping_threshold,
        # )

        self.trainer = GRPOTrainer(
                    model = self.model,
                    processing_class = self.actual_tokenizer,  # 使用純 tokenizer
                    train_dataset = dataset,
                    # eval_dataset = eval_dataset,
                    reward_funcs=[
                        hallucination_reward_function,
                    ],
                    args = GRPOConfig(
                        **dataclasses.asdict(self.train_args_cfg),
                        logging_dir=os.path.join(self.output_dir, "runs"),
                        output_dir = os.path.join(self.output_dir, "outputs")),
                    callbacks=[CustomLoggingCallback(logger=self.resource_logger)],  # 移除 early_stopping
                )
        
        # 保存原始的 _calculate_rewards 方法
        original_calculate_rewards = self.trainer._calculate_rewards
        
        def wrapped_calculate_rewards(self_trainer, inputs, prompts, completions, completion_ids_list):
            """計算 entropy 並注入到 reward function"""   
            start_time = time.time()         
            try:
                if hasattr(self_trainer.processing_class, 'tokenizer'):
                    tokenizer = self_trainer.processing_class.tokenizer
                else:
                    tokenizer = self_trainer.processing_class
                
                device = self_trainer.accelerator.device
                
                # 1. Tokenize prompts 獲取長度
                if isinstance(prompts[0], list):
                    prompt_texts = []
                    for p in prompts:
                        text = tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                        if text is None:
                            print(f"apply_chat_template 返回 None，使用空字符串")
                            text = ""
                        prompt_texts.append(text)
                else:
                    prompt_texts = prompts
                
                # 確保沒有 None 值傳入 processor
                prompt_texts = [text if text is not None else "" for text in prompt_texts]
                
                prompt_tokenized = tokenizer(
                    prompt_texts, return_tensors="pt", padding=True, 
                    truncation=True, max_length=self_trainer.max_prompt_length
                ).to(device)
                prompt_lengths = prompt_tokenized["attention_mask"].sum(dim=1).cpu()
                del prompt_tokenized
                torch.cuda.empty_cache()
                
                # 2. Tokenize 完整序列 (prompt + completion)
                if isinstance(prompts[0], list):
                    full_texts = []
                    for p, c in zip(prompts, completions):
                        combined = p + c if isinstance(c, list) else p + [{"role": "assistant", "content": c}]
                        text = tokenizer.apply_chat_template(
                            combined, tokenize=False, add_generation_prompt=False
                        )
                        if text is None:
                            print(f"apply_chat_template 返回 None，使用空字符串")
                            text = ""
                        full_texts.append(text)
                else:
                    full_texts = [p + c for p, c in zip(prompts, completions)]
                
                # 確保沒有 None 值傳入 processor
                full_texts = [text if text is not None else "" for text in full_texts]
                
                tokenized = tokenizer(
                    full_texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=self_trainer.max_prompt_length + self_trainer.max_completion_length
                ).to(device)
                
                # 3. 獲取 hidden states
                if not hasattr(self_trainer, '_autocast_dtype'):
                    self_trainer._autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                
                with torch.amp.autocast(device_type="cuda", dtype=self_trainer._autocast_dtype):
                    with torch.no_grad():
                        logits = self_trainer.model(
                            input_ids=tokenized["input_ids"],
                            attention_mask=tokenized["attention_mask"],
                            output_hidden_states=False
                        ).logits 

                batch_size = tokenized["input_ids"].shape[0]
                seq_len = tokenized["input_ids"].shape[1]

                entropy = torch.zeros(batch_size, seq_len, device='cpu', dtype=torch.float32)

                for i in range(batch_size):
                    log_probs = torch.nn.functional.log_softmax(logits[i], dim=-1)
                    sample_entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
                    entropy[i] = sample_entropy.squeeze(0).cpu()

                    del log_probs, sample_entropy
                    torch.cuda.empty_cache()
                
                del logits  
                torch.cuda.empty_cache()
                
                # 5. 創建 completion mask（只計算 completion 部分）
                batch_size, seq_len = entropy.shape
                completion_mask = torch.zeros_like(tokenized["attention_mask"], dtype=torch.float32, device='cpu')
                for i in range(batch_size):
                    prompt_len = prompt_lengths[i].item()-1
                    actual_len = tokenized["attention_mask"][i].sum().item()-1
                    completion_mask[i, prompt_len:actual_len] = 1.0
                
                # 6. 計算平均 entropy
                masked_entropy = entropy * completion_mask
                completion_token_counts = torch.clamp(completion_mask.sum(dim=1), min=1.0)
                avg_entropy = masked_entropy.sum(dim=1) / completion_token_counts
                
                print(f"平均 entropy: {avg_entropy.mean().item():.4f}")
                
                avg_entropy_list = avg_entropy.cpu().tolist()
                for i, example in enumerate(inputs):
                    example['avg_entropy'] = avg_entropy_list[i]
                    
                del tokenized
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"錯誤: {e}")
                import traceback
                traceback.print_exc()

            finally:
                # clean up
                gc.collect()
                torch.cuda.empty_cache()
                end_time = time.time()
                print(f"Entropy 計算時間: {end_time - start_time:.2f} 秒")
            
            return original_calculate_rewards(inputs, prompts, completions, completion_ids_list)
        
        # 使用 types.MethodType 將函數綁定為實例方法
        self.trainer._calculate_rewards = types.MethodType(wrapped_calculate_rewards, self.trainer)
      

    def train_and_save(self, resume_from_checkpoint=False):
        # 訓練前清理 GPU 記憶體
        torch.cuda.empty_cache()
        self.resource_logger.log_resource_usage("訓練前")
        
        try:
            trainer_stats = self.trainer.train(resume_from_checkpoint = resume_from_checkpoint)
        except torch.cuda.OutOfMemoryError as e:
            self.resource_logger.logger.error(f"CUDA OOM Error: {e}")
            # 清理記憶體並重新嘗試
            torch.cuda.empty_cache()
            gc.collect()
            raise e
        
        self.resource_logger.log_resource_usage("訓練後")
        self.model.save_pretrained(self.model_dir)
        self.actual_tokenizer.save_pretrained(self.model_dir)  # 保存純 tokenizer

        return trainer_stats
    
if __name__ == '__main__':
    # 清理 GPU 記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
    # 產生時間戳
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # 讓使用者輸入自訂名稱
    custom_name = "_halluciation_gemma3_12B_GRPO".strip()  
    # 組合 output_dir
    output_dir = os.path.join("train", f"{timestamp}_{custom_name}")

    trainer = UnslothTrainer(output_dir=output_dir)
    trainer.train_and_save()