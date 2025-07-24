import os
import json
import random
import json
import os
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any

from examples import get_examples


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def lower_keys(example):
    new_example = {}
    for key, value in example.items():
        if key != key.lower():
            new_key = key.lower()
            new_example[new_key] = value
        else:
            new_example[key] = value
    return new_example


EXAMPLES = get_examples()


def load_prompt(data_name, prompt_type, num_shots):
    if not num_shots:
        return []

    if data_name in ["gsm_hard", "svamp", "tabmwp", "asdiv", "mawps"]:
        data_name = "gsm8k"
    if data_name in ["math_oai", "hungarian_exam", "math-oai", "amc23", "phy1"]:
        data_name = "math"
    if data_name in ["sat_math"]:
        data_name = "mmlu_stem"
    if data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        data_name = "gaokao"

    if prompt_type in ["tool-integrated"]:
        prompt_type = "tora"

    return EXAMPLES[data_name][:num_shots]


PROMPT_TEMPLATES = {
    "direct": ("Question: {input}\nAnswer: ", "{output}", "\n\n"),
    "cot": ("Question: {input}\nAnswer: ", "{output}", "\n\n\n"),
    "pal": ("Question: {input}\n\n", "{output}", "\n---\n"),
    "tool-integrated": ("Question: {input}\n\nSolution:\n", "{output}", "\n---\n"),
    "self-instruct": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "tora": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "wizard_zs": (
        "### Instruction:\n{input}\n\n### Response: Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "platypus_fs": (
        "### Instruction:\n{input}\n\n### Response:\n",
        "{output}",
        "\n\n\n",
    ),
    "deepseek-math": (
        "User: {input}\nPlease reason step by step, "
        "and put your final answer within \\boxed{{}}.\n\nAssistant:",
        "{output}",
        "\n\n\n",
    ),
    "kpmath": (
        "User: Please reason step by step and put your final answer at the end "
        'with "The answer is: ".\n\n{input}\n\nAssistant:',
        "{output}",
    ),
    "jiuzhang": (
        "## Question\n{input}\n\n## Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_tora": (
        "## Question\n{input}\n\n## Code Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_nl": (
        "## Question\n{input}\n\n## Natural Language Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "mmiqc": (
        'Please solve the following problem and put your answer at the end with "The answer is: ".\n\n{input}\n\n',
        "{output}",
        "\n\n\n",
    ),
    "abel": (
        "Question:\n{input}\nAnswer:\nLet's think step by step.\n",
        "{output}",
        "\n\n",
    ),
    "shepherd": ("{input}\n", "{output}", "\n\n\n"),
    "qwen-boxed": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen25-math-cot-debug-1": (
        "<|im_start|>system\nPlease reason step by step.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen25-math-cot-debug-2": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen25-math-cot": (
        "<|im_start|>system\nYou are a helpful assistant participating in a math contest. The contest is ranked by pass@1 accuracy. Please be confident of outputting the answers that you think is most correct. Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\nQuestion: The pressure \\( P \\) exerted by wind on a sail varies jointly as the area \\( A \\) of the sail and the cube of the wind's velocity \\( V \\). When the velocity is \\( 8 \\) miles per hour, the pressure on a sail of \\( 2 \\) square feet is \\( 4 \\) pounds. Find the wind velocity when the pressure on \\( 4 \\) square feet of sail is \\( 32 \\) pounds. Let's think step by step and output the final answer within \\boxed{{}}.\nAnswer: To determine the wind velocity when the pressure on a sail of 4 square feet is 32 pounds, we start by expressing the relationship between pressure \\( P \\), area \\( A \\), and velocity \\( V \\) mathematically. Since the pressure varies jointly as the area of the sail and the cube of the wind's velocity, we can write:\n\n\\[ P = kA V^3 \\]\n\nwhere \\( k \\) is a constant of proportionality. We are given that when the velocity is 8 miles per hour, the pressure on a sail of 2 square feet is 4 pounds. We can use this information to find the value of \\( k \\). Substituting \\( P = 4 \\), \\( A = 2 \\), and \\( V = 8 \\) into the equation, we get:\n\n\\[ 4 = k \\cdot 2 \\cdot 8^3 \\]\n\nFirst, we calculate \\( 8^3 \\):\n\n\\[ 8^3 = 512 \\]\n\nSo the equation becomes:\n\n\\[ 4 = k \\cdot 2 \\cdot 512 \\]\n\nNext, we multiply 2 by 512:\n\n\\[ 2 \\cdot 512 = 1024 \\]\n\nThus, the equation simplifies to:\n\n\\[ 4 = k \\cdot 1024 \\]\n\nTo solve for \\( k \\), we divide both sides of the equation by 1024:\n\n\\[ k = \\frac{{4}}{{1024}} = \\frac{{1}}{{256}} \\]\n\nNow that we have the value of \\( k \\), we can use it to find the wind velocity when the pressure on a sail of 4 square feet is 32 pounds. Substituting \\( P = 32 \\), \\( A = 4 \\), and \\( k = \\frac{{1}}{{256}} \\) into the equation \\( P = kA V^3 \\), we get:\n\n\\[ 32 = \\frac{{1}}{{256}} \\cdot 4 \\cdot V^3 \\]\n\nFirst, we simplify the right side of the equation:\n\n\\[ \\frac{{1}}{{256}} \\cdot 4 = \\frac{{4}}{{256}} = \\frac{{1}}{{64}} \\]\n\nSo the equation becomes:\n\n\\[ 32 = \\frac{{1}}{{64}} V^3 \\]\n\nTo solve for \\( V^3 \\), we multiply both sides of the equation by 64:\n\n\\[ V^3 = 32 \\cdot 64 \\]\n\nNext, we calculate \\( 32 \\cdot 64 \\):\n\n\\[ 32 \\cdot 64 = 2048 \\]\n\nThus, the equation simplifies to:\n\n\\[ V^3 = 2048 \\]\n\nTo find \\( V \\), we take the cube root of both sides of the equation:\n\n\\[ V = \\sqrt[3]{{2048}} = 12 \\]\n\nTherefore, the wind velocity when the pressure on 4 square feet of sail is 32 pounds is \\(\\boxed{{12}}\\) miles per hour.\n\n\nQuestion: {input}\nAnswer:<|im_end|>\n"
        # "<|im_start|>assistant\n Question: {input} \n Answer: To",
        "<|im_start|>assistant\nTo",
        "{output}",
        "\n\n",
    ),
    "qwen25-math-cot-assistant": (
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}\n",
        "\n\n",
    ),
    "qwen25-math-cot-tool": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "mathstral": (
        "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "{output}",
        "\n\n",
    ),
    "internlm-math-fs": ("Question:{input}\nAnswer:", "{output}", "\n"),
    "internlm-math-chat": (
        "<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "mistral": (
        "[INST] {input}[/INST]",
        "{output}",
        "\n\n",
    ),
    "numina": ("### Problem: {input}\n### Solution:", " {output}", "\n\n"),
    "o1_cot": (
        '[Round 0] USER:\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}. ASSISTANT:\n',
        "{output}",
        "\n\n"
    ),
    "deepseek-r1": (
        'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {input}. Assistant:',
        "{output}",
        "\n\n"
    ),
    "llama": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 17 Apr 2025\n\n"
        " <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "{output}",
        "\n\n",
    ),
    "deepseek-distill": (
        "<｜begin▁of▁sentence｜><｜User｜>{input}<｜Assistant｜><think>\n",
         "{output}",
        "\n\n",
    )
}


def construct_prompt(example, data_name, args):
    if args.adapt_few_shot and data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        demos = load_prompt(data_name, args.prompt_type, 5)
    else:
        demos = load_prompt(data_name, args.prompt_type, args.num_shots)
    # print("############################## This is good.")
    # for q, a in demos:
    #     print("########################### Demo (q): ", q)
    #     print("###########################")
    #     print("########################### Demo (a): ", a)
    prompt_type = args.prompt_type
    if prompt_type == "platypus_fs":
        prompt_type = "cot"
    if prompt_type == "tool-integrated":
        prompt_type = "tora"
    prompt_temp = PROMPT_TEMPLATES[args.prompt_type]

    splitter = prompt_temp[2]
    input_template, output_template, splitter = (
        prompt_temp[0],
        prompt_temp[1],
        prompt_temp[2],
    )
    if args.prompt_type == "qwen25-math-cot":
        # Hotfix to support putting all demos into a single turn
        demo_prompt = splitter.join([q + "\n" + a for q, a in demos])
    elif args.prompt_type == "qwen25-math-cot-tool" :
        demo_prompt = "".join([f"Question: {q}\nSolution: {a}\n\n" for q, a in demos])
    else:
        demo_prompt = splitter.join(
            [
                input_template.format(input=q) + output_template.format(output=a)
                for q, a in demos
            ]
        )
    context = input_template.format(input=example["question"])
    if len(demo_prompt) == 0 or (
        args.adapt_few_shot and example["gt_ans"] not in ["A", "B", "C", "D", "E"]
    ): 
        full_prompt = context
    else:
        if args.prompt_type == "qwen25-math-cot":
            # Hotfix to supportting put all demos into a single turn
            full_prompt = demo_prompt + splitter + example["question"]
            full_prompt = input_template.format(input=full_prompt)
        elif args.prompt_type == "qwen25-math-cot-tool":
            full_prompt = demo_prompt + splitter + "Question: " + example["question"]
            full_prompt = input_template.format(input=full_prompt)
        elif args.prompt_type == "qwen25-math-cot-assistant":
            full_prompt = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n" + demo_prompt+input_template.format(input=example["question"])
        else:
            full_prompt = demo_prompt + splitter + context

    if args.prompt_type == "platypus_fs":
        full_prompt_temp = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        )
        full_prompt = full_prompt_temp.format(instruction=full_prompt)

    if prompt_type == "tora":
        full_prompt = (
            """Integrate step-by-step reasoning and Python code to solve math problems using the following guidelines:

- Analyze the question and write functions to solve the problem; the function should not take any arguments.
- Present the final result in LaTeX using a `\boxed{}` without any units.
- Utilize the `pi` symbol and `Rational`` from Sympy for $\pi$ and fractions, and simplify all fractions and square roots without converting them to decimal values.

Here are some examples you may refer to:

---

"""
            + full_prompt
        )

    return full_prompt.strip(" ")  # important!


key_map = {
    "gt": "Ground Truth",
    "pred": "Prediction",
    "gt_cot": "Reference CoT",
    "score": "Score",
}


def show_sample(sample, print_all_preds=False):
    print("==" * 20)
    for key in ["idx", "type", "level", "dataset"]:
        if key in sample:
            # capitalize
            print("{}: {}".format(key[0].upper() + key[1:], sample[key]))
    print("Question:", repr(sample["question"]))
    if "code" in sample:
        if print_all_preds:
            for code in sample["code"]:
                print("-" * 20)
                print("code:", code)
            print("Execution:", sample["report"])
        else:
            print("Solution:\n", sample["code"][0])
            print("Execution:", sample["report"][0])
    if "pred" in sample:
        print("Prediction:", repr(sample["pred"][0]))
    for key in ["gt", "score", "unit", "gt_cot"]:
        if key in sample:
            _key = key_map.get(key, key)
            print("{}: {}".format(_key, repr(sample[key])))
    print()
