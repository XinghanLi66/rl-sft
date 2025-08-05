import argparse
import os
import random
import time
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

import wandb

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


os.environ.setdefault("NCCL_TIMEOUT", "2700")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "2700")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen2.5-Math-7B', help='Model name')
    parser.add_argument('--model_path', type=str, default=None, help='Local model path')
    parser.add_argument('--train_data', type=str, default='dataset/1shot_rlvr/pi1_r1280.parquet', help='Training data file path')
    parser.add_argument('--save_root', type=str, default=None, help='Checkpoint save root directory')
    parser.add_argument('--effective_batch', type=int, default=64, help='Global batch size')
    parser.add_argument('--micro_batch_size', type=str, default=2, help='Micro batch size or "auto"')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature coefficient')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--log_steps', type=int, default=1, help='Logging step interval')
    parser.add_argument('--save_steps', type=int, default=1, help='Checkpoint saving step interval')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum training steps')
    parser.add_argument('--sample_temp', type=float, default=0.5, help='Generation temperature parameter')
    parser.add_argument('--run_name', type=str, default=None, help='Experiment run name')
    parser.add_argument('--wandb_project', type=str, default='entropy-maximization-ft', help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='W&B run name')
    parser.add_argument('--seed', type=int, default=15, help='Random seed')
    return parser.parse_args()

class FTDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]

def custom_collate(batch):
    return {"input": [item["input"] for item in batch]}

def get_optimal_micro_batch_size(model_name: str, world_size: int = 1) -> int:
    model_configs = {
        "1.5B": {"base_batch": 4, "keywords": ["1.5B", "1B"]},
        "2B": {"base_batch": 4, "keywords": ["2B"]},
        "3B": {"base_batch": 2, "keywords": ["3B"]},
        "7B": {"base_batch": 2, "keywords": ["7B"]},
        "8B+": {"base_batch": 1, "keywords": ["8B", "9B", "10B", "11B", "12B", "13B", "14B"]},
    }
    model_name_upper = model_name.upper()
    detected = next((cfg for cfg in model_configs.values() if any(k in model_name_upper for k in cfg["keywords"])), None)
    base_batch = detected["base_batch"] if detected else 2
    if world_size > 1:
        return min(base_batch + 1, int(base_batch * 1.5))
    return base_batch

COT_TEMPLATES = {
    "qwen25-math-cot": """
<|im_start|>system
{{ system_prompt }}<|im_end|>
<|im_start|>user
Question: {{ example_question }}
Answer: {{ example_answer }}


Question: {{ question }}
Answer:<|im_end|>
<|im_start|>assistant
Question: {{ question }}
Answer: To
"""
}

cnt = 0

COT_TEMPLATES = {
    "qwen25-math-cot": r"""
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'Please reason step by step, and put your final answer within \\boxed{}.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nPlease reason step by step.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- elif message.role == "generation_prompt" %}
        {{- message.content }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""
}



cnt = 0

def apply_chat_template(tokenizer, problem: str, template_key="qwen25-math-cot") -> str:
    global cnt
    # print(f"tokenizer.chat_template", tokenizer.chat_template)

    tokenizer.chat_template = COT_TEMPLATES[template_key]

    user_content = "Question: The pressure \\( P \\) exerted by wind on a sail varies jointly as the area \\( A \\) of the sail and the cube of the wind's velocity \\( V \\). When the velocity is \\( 8 \\) miles per hour, the pressure on a sail of \\( 2 \\) square feet is \\( 4 \\) pounds. Find the wind velocity when the pressure on \\( 4 \\) square feet of sail is \\( 32 \\) pounds. Let's think step by step and output the final answer within \\boxed{{}}.\nAnswer: To solve the problem, we start by writing the mathematical relationship for the pressure \\( P \\):\n\\[ P = k \\cdot A \\cdot V^3 \\]\nwhere \\( k \\) is a constant. We need to find \\( k \\) using the given information:\n\\[ 4 = k \\cdot 2 \\cdot 8^3 \\]\nSolving for \\( k \\):\n\\[ 4 = k \\cdot 2 \\cdot 512 \\]\n\\[ 4 = 1024k \\]\n\\[ k = \\frac{{4}}{{1024}} \\]\n\\[ k = \\frac{{1}}{{256}} \\]\nNow we use this value of \\( k \\) to find the velocity \\( V \\) when the pressure \\( P \\) on 4 square feet of sail is 32 pounds:\n\\[ 32 = \\frac{{1}}{{256}} \\cdot 4 \\cdot V^3 \\]\n\\[ 32 = \\frac{{V^3}}{{64}} \\]\n\\[ 32 \\cdot 64 = V^3 \\]\n\\[ 2048 = V^3 \\]\n\\[ V = \\sqrt[3]{{2048}} \\]\n\\[ V = 12.8 \\]\nThus, the wind velocity is \\( \\boxed{{12.8}} \\) miles per hour.\n\n\nQuestion: " + problem + "\nAnswer:"

    generation_prompt_content = "<|im_start|>assistant\n Question: " + problem + " \n Answer: To"

    full_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content},
         {"role": "generation_prompt", "content": generation_prompt_content}],
        tokenize=False,
        # add_generation_prompt=True,
    )

    cnt += 1
    if cnt == 1:
        print(f"################### user_content ###################\n\n{user_content}\n")
        print(f"################### generation_prompt_content ###################\n\n{generation_prompt_content}\n")
        print(f"################### full input ###################\n\n{full_input}\n")

    return full_input

def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    micro_bs = int(args.micro_batch_size)
    eff_bs = args.effective_batch
    accum_steps = max(1, eff_bs // (micro_bs * world_size))
    temp = args.temperature
    lr = args.learning_rate

    save_root = args.save_root or "checkpoints"
    save_root = f"{save_root}/{args.model_name}/{args.run_name}" if args.run_name else f"checkpoints/{args.model_name}"

    ds_config = {
        "train_micro_batch_size_per_gpu": micro_bs,
        "train_batch_size": eff_bs,
        "gradient_accumulation_steps": accum_steps,
        "bf16": {"enabled": True},
        "zero_optimization": {
                              "stage": 2, 
                              "offload_optimizer": {"device": "cpu"}, 
                              "offload_param": {"device": "none"}
                             },
        "gradient_clipping": 1.0,
    }
    accelerator = Accelerator(mixed_precision="bf16", 
                              gradient_accumulation_steps=accum_steps, 
                              deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=ds_config))
    print = accelerator.print

    model_path = args.model_path or f"/volume/pt-train/models/{args.model_name}"
    config = AutoConfig.from_pretrained(model_path)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config)
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.run_name or args.wandb_name or args.model_name, config=vars(args))

    df = pd.read_parquet(args.train_data)
    train_data = [{"input": apply_chat_template(tokenizer, p)} for p in df["problem"].dropna().tolist()]
    train_loader = DataLoader(FTDataset(train_data), batch_size=micro_bs, shuffle=True, collate_fn=custom_collate)

    optimizer = AdamW(model.parameters(), lr=lr)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    prev_logits = None
    model.train()
    
    ## at most one epoch?
    for step, batch in enumerate(train_loader, start=1):
        if step > args.max_steps:
            print(f"Exceed max step {args.max_steps}, training stopped.")
            break
        
        with accelerator.accumulate(model):
            enc = tokenizer(batch["input"], 
                            return_tensors="pt", 
                            padding="longest", 
                            truncation=True, 
                            max_length=2048).to(accelerator.device)
            
            with torch.no_grad():
                gen_ids = accelerator.unwrap_model(model).generate(**enc, 
                                                                   max_new_tokens=512, 
                                                                   do_sample=True, 
                                                                   top_p=0.95, 
                                                                   temperature=args.sample_temp, 
                                                                   synced_gpus=True, 
                                                                   repetition_penalty=1.15,
                                                                   pad_token_id=tokenizer.pad_token_id, 
                                                                   use_cache=False)
                
            seq = torch.cat([enc.input_ids, gen_ids[:, enc.input_ids.shape[1]:]], dim=1)[:, :4096]
            pad_mask = seq.ne(tokenizer.pad_token_id)
            prompt_len = pad_mask[:, :enc.input_ids.shape[1]].sum(-1)
            token_idx = torch.arange(seq.size(1), device=seq.device)
            gen_mask = (token_idx.unsqueeze(0) >= prompt_len.unsqueeze(1)) & pad_mask

            logits = model(seq, attention_mask=pad_mask).logits
            probs = F.softmax(logits / temp, dim=-1)
            H_tok = -(probs * torch.log(probs + 1e-12)).sum(-1)
            loss = (H_tok * gen_mask).sum() / gen_mask.sum().clamp_min(1)

            prev_logits = logits.detach()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if accelerator.is_main_process:
            if step % args.log_steps == 0:
                print(f"Step {step} | loss={loss.item():.6f}")
                wandb.log({"step": step, "loss": loss.item()})
                
            if step % args.save_steps == 0:
                ckpt = Path(save_root) / f"step_{step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(ckpt, safe_serialization=True)
                tokenizer.save_pretrained(ckpt)
                print(f"Checkpoint saved to {ckpt}")

    if accelerator.is_main_process:
        final = Path(save_root) / "final"
        final.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(final, safe_serialization=True)
        tokenizer.save_pretrained(final)
        print(f"Final checkpoint saved to {final}")
        wandb.finish()

if __name__ == "__main__":
    main()
