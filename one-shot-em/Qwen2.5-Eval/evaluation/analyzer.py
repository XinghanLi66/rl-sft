import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

datasets = ['amc23x8', 'math500', 'minerva_math', 'olympiadbench']
datasets_06 = ['amc23x8', 'aime25x8']

def extract_scores_from_json(json_file):
    """Extract scores from a JSON evaluation result file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    try:
        return data["total_acc"]
    except KeyError:
        raise ValueError(f"Key 'scores' not found in JSON file: {json_file}")

def extract_scores_from_path(path: Path):
    json_files = glob.glob(str(path / '*.json'))
    if not json_files:
        print(f"No JSON files found in path: {path}")
        return None
        # raise ValueError(f"No JSON files found in path: {path}")
    json_file = json_files[0]  # Assuming we only want the first JSON file
    return extract_scores_from_json(json_file)

def sort_checkpoint(x):
    """Sort function for checkpoint names.
    Extracts the numeric part from checkpoint names like 'checkpoint-100' and returns it as an integer.
    """
    try:
        # Extract the number after 'checkpoint-'
        return int(x.split('_')[1])
    except (IndexError, ValueError):
        # If the format is unexpected, return 0 to put it at the start
        return 1000000

def main():
    # Create results directory if it doesn't exist
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Find all checkpoint directories
    # checkpoint_base = Path('/homes/gws/lxh22/rl-sft/one-input-sft/save/Qwen2.5-Math-1.5B-filter1') 
    checkpoint_base = Path('/local1/lxh/save/em/Qwen2.5-Math-1.5B/prompted_one_shot_1.5b_t0.5') 
    # checkpoint_base = Path('/homes/gws/lxh22/rl-sft/one-shot-em/checkpoints/Qwen2.5-Math-1.5B/one_shot_1.5b_t0.5') ## change this

    steps = [5, 10, 15, 20]
    all_scores = []
    
    for step in steps:
        print(f"Processing step: {step}")
        step_str = f'step_{step}'
        current_score = {'checkpoint': step_str}
        for dataset in datasets:
            extracted = extract_scores_from_path(checkpoint_base / step_str / f'temp00/{dataset}')
            if extracted: 
                current_score[dataset] = extracted

        for dataset in datasets_06:
            extracted = extract_scores_from_path(checkpoint_base / step_str / f'temp06/{dataset}')
            if extracted: 
                current_score[dataset + '-t0.6'] = extracted

        # extracted = extract_scores_from_path(checkpoint_base / step_str / 'temp01/amc-eval/amc23x8')
        # if extracted:
        #     current_score['amc-t0'] = extracted
        # extracted = extract_scores_from_path(checkpoint_base / step_str / 'temp03/amc-eval/amc23x8')
        # if extracted:
        #     current_score['amc'] = extracted        
        # extracted = extract_scores_from_path(checkpoint_base / step_str / 'temp04/amc-eval/amc23x8')
        # if extracted:
        #     current_score['amc-to'] = extracted

        all_scores.append(current_score)

    # Convert to DataFrame
    df = pd.DataFrame(all_scores)
    
    # Sort the DataFrame by checkpoint number
    df = df.sort_values(by='checkpoint', key=lambda col: [sort_checkpoint(x) for x in col])
    
    # Save raw data
    df.to_csv(results_dir / 'checkpoint_scores.csv', index=False)
    
    ## Amc line styles
    # amc_styles = {'amc23x8': '-', 'amc-t0.6': ':', 'amc-to': '--', 'amc-to-t0.5': '-.'}

    # Create line plot
    plt.figure(figsize=(12, 6))
    for column in df.columns:
        if column != 'checkpoint':
            if not column.endswith('-t0.6'):
                plt.plot(df['checkpoint'], df[column], marker='o', label=column)
            else:
                plt.plot(df['checkpoint'], df[column], marker='o', label=column, \
                         linestyle=':', color='purple')
    
    plt.title('Evaluation Scores Across Checkpoints')
    plt.xlabel('Checkpoint')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(results_dir / 'checkpoint_scores.png')
    plt.close()

if __name__ == '__main__':
    main()