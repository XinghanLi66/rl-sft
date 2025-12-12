import os
import json
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def extract_scores_from_json(json_file):
    """Extract scores from a JSON evaluation result file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if "acc" in data:
        return data["acc"]
    else:
        raise ValueError(f"No 'acc' key found in {json_file}")

def sortfunc(x):
    if x.name == "final":
        return 100000
    else:
        return int(x.name.split('_')[-1])

def main():
    parser = argparse.ArgumentParser()
    ## also use -t for --task_name, -c for --checkpoint_base
    parser.add_argument('-t', '--task_name', type=str, default='pi1-r16k-ofrl-qwen-2.5-math-1.5b-20steps')
    parser.add_argument('-c', '--checkpoint_base', type=str, default='/local1/lxh/save/offline_grpo/1.5b_pi1_ofrl')
    args = parser.parse_args()
    task_name = args.task_name
    checkpoint_base = Path(args.checkpoint_base)

    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    checkpoint_dirs = [x for x in checkpoint_base.iterdir() if x.is_dir()]
    checkpoint_dirs = sorted(checkpoint_dirs, key=sortfunc)
    
    # Collect scores for each checkpoint
    all_scores = []
    if "1.5b" in task_name or "1.5B" in task_name:
        all_scores.append(
            {
                "checkpoint": "global_step_0",
                "math500": 35.8,
                "minerva_math": 10.3,
                "olympiadbench": 22.8,
                "amc23x8-t06": 32.8,
                "aime25x8-t06": 2.9
            }
        )
    elif "7b" in task_name or "7B" in task_name:
        all_scores.append(
            {
                "checkpoint": "global_step_0",
                "math500": 50.4,
                "minerva_math": 12.5,
                "olympiadbench": 16.4,
                "amc23x8-t06": 30.9,
                "aime25x8-t06": 2.1
            }
        )
    else:
        raise NotImplementedError("Lack performance of this base model.")
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_name = checkpoint_dir.name
        print("checkpoint_name: ", checkpoint_name)
        scores = {}
        no_score_flag = False

        for dataset in ['math500', 'minerva_math', 'olympiadbench']:
            json_files = glob.glob(str(checkpoint_dir / "temp00" / dataset / '*metrics.json'))
            if len(json_files) == 1:
                json_file = json_files[0]
                scores[dataset] = extract_scores_from_json(json_file)                
            elif len(json_files) == 0:
                no_score_flag = True
            else:
                raise ValueError(f"Expected 1 metrics.json file, found {len(json_files)} for checkpoint {checkpoint_name}")

        if no_score_flag:
            print(f"Skipping checkpoint {checkpoint_name} due to missing scores.")
            continue

        for dataset in ['amc23x8', 'aime25x8']: 
            json_files = glob.glob(str(checkpoint_dir / "temp06" / dataset / '*metrics.json'))
            if len(json_files) != 1:
                raise ValueError(f"Expected 1 metrics.json file, found {len(json_files)} for checkpoint {checkpoint_name}")
            json_file = json_files[0]
            scores[dataset + '-t06'] = extract_scores_from_json(json_file)

        scores['checkpoint'] = checkpoint_name
        all_scores.append(scores)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_scores)
    
    # Save raw data
    df.to_csv(results_dir / f'{task_name}.csv', index=False)

    # Create line plot, also plot the average score
    datasets = ['math500', 'minerva_math', 'olympiadbench', 'amc23x8-t06', 'aime25x8-t06']
    plt.figure(figsize=(12, 6))
    for dataset in datasets:
        marker = 's' if 't06' in dataset else 'o'
        if dataset in df.columns:
            plt.plot(df['checkpoint'], df[dataset], marker=marker, label=dataset)

    # plot the average score
    plt.plot(df['checkpoint'], df.drop(columns=['checkpoint']).mean(axis=1), marker='o', label='Average')

    plt.title('Evaluation Scores Across Checkpoints')
    plt.xlabel('Checkpoint')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(results_dir / f'{task_name}.png')
    plt.close()

    # give me the highest score for each dataset
    for dataset in datasets:
        print(f"Highest score for {dataset}: {df[dataset].max()}")

    # give me the checkpoint name for the highest score for each dataset
    for dataset in datasets:
        print(f"Checkpoint name for highest score for {dataset}: {df[df[dataset] == df[dataset].max()]['checkpoint'].values[0]}")
    
    # give me the checkpoint name for the highest average score (remember to exclude the 'checkpoint' column)
    print(f"Checkpoint name for highest average score: {df.iloc[df.drop(columns=['checkpoint']).mean(axis=1).idxmax()]['checkpoint']}")

if __name__ == '__main__':
    main() 