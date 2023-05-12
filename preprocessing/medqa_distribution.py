import os
import glob
import json
import argparse

ANSWER_LIST = ['A', 'B', 'C', 'D']

def read_jsonl(f: str) -> list[dict]:
    """Read a jsonl file."""
    with open(f, 'r') as f:
        return [json.loads(line) for line in f.readlines()]

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_dir', type=str, default='data/MedQA/')
    argparser.add_argument('--condition_to_analyze', type=str, required=True)
    argparser.add_argument('--output_dir', type=str, default='output/MedQA/')
    args = argparser.parse_args()

    # Get all files
    files = glob.glob(args.input_dir + '*.jsonl')

    all_data = []
    for f in files:
        all_data.extend(read_jsonl(f))

    to_check_for = args.condition_to_analyze.lower()
    found, is_answer = [], []
    for qa_dict in all_data:
        is_in_arr = [to_check_for in x.lower() for x in qa_dict['options'].values()]
        answer_index = ANSWER_LIST.index(qa_dict['answer_idx'])


                    # We know the element we are looking for is in here!
        if any(is_in_arr) and (is_in_arr.index(True) == answer_index):
            is_answer.append(qa_dict)
            
        elif any(is_in_arr):
            found.append(qa_dict)

    # Print our results and then save to the file        
    print(f'Found {len(is_answer)} answers and {len(found)} questions with the condition {args.condition_to_analyze}')

    # First check if it exists and that's fine if it does.
    if os.makedirs(args.output_dir, exist_ok=True):
        print('Output directory already exists. Continuing...')

    # Then save to file
    json.dump(is_answer, open(os.path.join(args.output_dir, f'is_answer_{args.condition_to_analyze}.json'), 'w'), indent=4)
    json.dump(found, open(os.path.join(args.output_dir, f'found_{args.condition_to_analyze}.json'), 'w'), indent=4)