import os
import argparse
import glob 
import pandas as pd
from src.healer_cases.healer_cases_statistics import graph_kendalltaus

# python src/combined_healer_cases.py --input_path output/figures/healer_cases/Dyspnea_3.0_cases --output_path ./output/figures/healer_cases/combined_dyspnea/
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input_path', type=str, required=True)
    args.add_argument('--output_path', type=str, required=True)
    args = args.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # Load all the csvs
    csvs = glob.glob(f'{args.input_path}/*.csv')
    dfs = [pd.read_csv(x) for x in csvs if 'kendall_tau' in x]
    df = pd.concat(dfs)

    if 'Unnamed: 0' in df:
        df = df.drop(columns=['Unnamed: 0'])

    # Graph the kendall taus
    graph_kendalltaus(df, args.output_path, 'combined')
    df.to_csv(f'{args.output_path}/kendall_tau_combined_cases.csv', index=False)

