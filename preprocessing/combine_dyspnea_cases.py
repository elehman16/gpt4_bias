import os
import re
import pickle
import argparse
from glob import glob
from typing import Tuple, List

def load_pickle_file(f: str):
    """Load the pickle file. """
    with open(f, 'rb') as f:
        return pickle.load(f)
    
def load_joint_files(path_dir_1: str, path_dir_2: str) -> List[Tuple[dict]]:
    """Load the joint files. """
    files_dir_1 = glob(f"{path_dir_1}/*.pkl")
    files_dir_2 = glob(f"{path_dir_2}/*.pkl")

    # Make this awful assumption -- L O L.
    files_dir_1 = sorted(files_dir_1)
    files_dir_2 = sorted(files_dir_2)
    assert len(files_dir_1) == len(files_dir_2)

    return list(zip(files_dir_1, files_dir_2))
    
def combine_healer_cases_results(res1: dict, res2: dict) -> dict:
    """Combine two healer case dictionaries. """
    new_dict_to_construct = {}
    for key in res1:
        if key == 'Real_DDx': 
            new_dict_to_construct[key] = res1[key] 
        else:   
            new_dict_to_construct[key] = res1[key] + res2[key]

    return new_dict_to_construct    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_dir_1', type=str, default="./data/healer_cases/dyspnea_cases_2.0/")
    argparser.add_argument('--input_dir_2', type=str, default="./data/healer_cases/dyspnea_cases_3.0/")
    argparser.add_argument('--output_dir', type=str, default="./data/healer_cases/dyspnea_cases_4.0/")
    args = argparser.parse_args()

    # Make the output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    files = load_joint_files(args.input_dir_1, args.input_dir_2)
    
    for (f1, f2) in files:
        res1 = load_pickle_file(f1)
        res2 = load_pickle_file(f2)

        # Combine the results
        combined_res = combine_healer_cases_results(res1, res2)

        # Save the results
        # 1. Get the case # -> 
        # 2. Save the results
        case_num = re.search(r"case_(\d+)", f1).group(1)
        with open(f"{args.output_dir}/case_{case_num}.pkl", 'wb') as f:
            pickle.dump(combined_res, f)
