import json
import argparse
import pandas as pd
from utils import run_prompts

def construct_prompts(df: pd.DataFrame) -> list[list[dict]]:
    """Construct the prompts for the given dataframe."""
    prompts = []
    for _, row in df.iterrows():
        query = [
            {"role": "system", "content": row['system']},
            {"role": "user", "content": row['prompt']},
        ]

        prompts.append(query)

    return prompts


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_file', type=str, required=True)
    argparser.add_argument('--temperature', type=float, default=0.7)
    argparser.add_argument('--max_tokens', type=int, default=500)
    argparser.add_argument('--num_samples', type=int, default=50)
    argparser.add_argument('--n_workers', type=int, default=4)
    argparser.add_argument('--output_file', type=str, required=True)
    args = argparser.parse_args()

    # Read the dataframe
    df = pd.read_csv(args.input_file)
    prompts = construct_prompts(df)

    gpt4_outputs = run_prompts(
        prompts=prompts, 
        num_samples=args.num_samples, 
        temperature=args.temperature, 
        max_tokens=args.max_tokens, 
        n_workers=args.n_workers
    )

    json.dump(gpt4_outputs, open(args.output_file, 'w'))
    