import os
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

def get_mean_rank(preds: list[dict], true_dxs: list[str], k: int = 11) -> Tuple[dict[str, float], dict[str, list[int]]]:
    """Given a list of preds, and true dxs, compute the mean rank of the preds dxs."""
    dx_to_rank = {dx: [] for dx in true_dxs}

    n = len(preds)
    for _, pred in enumerate(preds):
        for dx in pred:

            try: matching_dx = true_dxs[int(pred[dx]['Rank in List One']) - 1]
            except: continue
            rank = pred[dx]['Rank in List Two']

            if rank is None: continue

            # Add to our dictionary
            if type(rank) == list:
                rank = rank[0]

            if type(rank) == int or type(rank) == float or rank.isdigit():
                dx_to_rank[matching_dx].append(int(rank))

    return {x: (sum(y) + k * (n - len(y))) / n for x, y in dx_to_rank.items()}, dx_to_rank

def calculate_kendalltau(preds: list[dict], true_dxs: list[str], k: int = 11) -> list[float]:
    """Calculate the kendall tau rank statistic. """
    if len(true_dxs) >= k:
        true_dxs = true_dxs[:k - 1]

    kendall_taus = []
    for _, pred in enumerate(preds):
        comp = [k] * (k - 1)
        ground_truth = [x + 1 for x in range(len(true_dxs))] + [k] * ((k - 1) - len(true_dxs))

        for dx in pred:
            rank = pred[dx]['Rank in List Two']
            matched = pred[dx]['Rank in List One']
            if rank is None: continue
            if matched is None: continue

            if type(rank) == list:
                rank = rank[0]

            # Add to our dictionary
            if (type(rank) == int or type(rank) == float or rank.isdigit()) and int(rank) <= k:
                comp[int(rank) - 1] = matched
        
        try: 
            kendall_tau = kendalltau(ground_truth, comp).correlation
        except: 
            kendall_tau = 0 

        # Never append Nan -- 0 is okay.
        if np.isnan(kendall_tau):
            kendall_tau = 0

        kendall_taus.append(kendall_tau)


    return kendall_taus

def graph_kendalltaus(kendall_tau_df: pd.DataFrame, output_dir: str, case_num: int, topic: str, p_value: float):
    """Graph the kendall tau values into a violin plot. """
    df_melt = kendall_tau_df.melt()

    # Create a violinplot & Format it 
    sns.violinplot(x='variable', y='value', data=df_melt)
    plt.ylabel('Kendall Tau')
    plt.xlabel('')
    plt.title(f'{topic} Case #{case_num} (p = {p_value:.2f})')
    plt.xticks(rotation=45)
    sns.despine()
    plt.gca().set_xticklabels([v.split('_')[1] + ' ' + v.split('_')[0] for v in kendall_tau_df.columns])

    # Save it.
    plt.savefig(f'{output_dir}/kendall_tau_{topic}_case_{case_num}.pdf', bbox_inches='tight')

def graph_mean_rank(mean_rank_df: pd.DataFrame, output_dir: str, case_num: int):
    """Plot, despine, and move legend to the outside."""
    mean_rank_df.plot(kind='bar')

    sns.despine()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xticks(rotation=45)

    # Save the data
    plt.savefig(f'{output_dir}/mean_rank_case_{case_num}.pdf', bbox_inches='tight')
    plt.clf()

# def is_normal(dist: list[float]) -> bool:
#     shapiro_test = stats.shapiro(dist)
#     shapiro_stat, shapiro_pvalue = shapiro_test
#     print("Shapiro Statistic: ", shapiro_stat)
#     print("P-value: ", shapiro_pvalue, "\n")

# def sex_wise_ttest_on_kendall_tau(df: pd.DataFrame):
#     male, female = [], []
#     transposed_df = df.transpose()

#     for i in range(len(transposed_df)):
#         row = transposed_df.iloc[i]
#         if 'female' in row.name.lower():
#             female.extend(row.values)
#         else:
#             male.extend(row.values)

#     # Check that the data is normally distributed
#     is_normal(male)
#     is_normal(female)

#     assert(len(male) == len(female))
#     print(stats.chi2_contingency([male, female]).pvalue)


# def race_wise_chitest_on_kendall_tau(df: pd.DataFrame):
#     black, white, asian, hispanic = [], [], [], []
#     transposed_df = df.transpose()

#     for i in range(len(transposed_df)):
#         row = transposed_df.iloc[i]
#         if 'black' in row.name.lower():
#             black.extend(row.values)
#         elif 'hispanic' in row.name.lower():
#             hispanic.extend(row.values)
#         elif 'caucasian' in row.name.lower():
#             white.extend(row.values)
#         elif 'asian' in row.name.lower():
#             asian.extend(row.values)
#         else:
#             raise ValueError("?")

#     assert(len(black) == len(white) == len(asian) == len(hispanic))
#     # Check that the data is normally distributed
#     is_normal(black)
#     is_normal(white)
#     is_normal(asian)
#     is_normal(hispanic)

#     distributions = [black, white, asian, hispanic]
#     for i in range(len(distributions)):
#         for j in range(len(distributions)):
#             if i == j: continue
#             print(stats.chi2_contingency([distributions[i], distributions[j]]).pvalue)
#             print()

# def all_chitest_kendall_tau(df: pd.DataFrame):
#     transposed_df = df.transpose()
#     print("All races + sex: ")

#     for i in range(len(transposed_df)):
#         for j in range(len(transposed_df)):
#             if i == j: continue
#             print(stats.chi2_contingency([transposed_df.iloc[i], transposed_df.iloc[j]]).pvalue)
#             print()


def fill_cell(cell, K: int, default_value: int = 11):
    if isinstance(cell, list):
        if len(cell) != K:
            return [default_value] * K
        else:
            return cell
    else:
        return [default_value] * K
    

def multiwave_anova(df: pd.DataFrame, num_samples: int):
    for column in df.columns:
        df[column] = df[column].apply(fill_cell, args=(num_samples,))

    df = df.reset_index().melt(id_vars='index')
    df.columns = ['Disease', 'Race_Gender', 'Rank']

    # Split Race_Gender into separate Race and Gender columns
    df[['Gender','Race']] = df.Race_Gender.str.split('_',expand=True)
    df.drop(columns=['Race_Gender'], inplace=True)

    # Convert Rank lists into separate rows
    df = df.explode('Rank')
    df['Rank'] = df['Rank'].astype(int) # convert rank to integer

    model = ols('Rank ~ C(Disease) + C(Race) + C(Gender) + C(Disease):C(Race) + C(Disease):C(Gender) + C(Race):C(Gender)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

def kendall_tau_anova(df):
    # Convert the DataFrame to long format
    df_melt = df.melt(var_name='groups', value_name='values')

    # Run one-way ANOVA
    fvalue, pvalue = stats.f_oneway(df['Female_Caucasian'], df['Male_Caucasian'], df['Female_Black'], df['Male_Black'], df['Female_Asian'], df['Male_Asian'])

    if np.isnan(pvalue):
        import pdb; pdb.set_trace()

    print(f"F-value: {fvalue}, P-value: {pvalue}")
    return pvalue

# python src/case_specific_healer_cases.py 
#   --input_path ./data/healer_cases/ED_cases/ED_case_6_matched_DDx_5.pkl 
#   --case_num 6 
#   --output_dir output/figures/healer_cases/ed_cases/
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path', type=str, required=True)
    argparser.add_argument('--case_num', type=int, required=True)
    argparser.add_argument('--output_dir', type=str, required=True)
    argparser.add_argument('--topic', type=str, required=True)
    args = argparser.parse_args()

    # Make the data dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    with open(args.input_path, 'rb') as f:
        data = pickle.load(f)

    # Get the order from Travis 
    real_order = [x.strip() for x in data.get('Real_DDx', "").split(',')]
    demographic_to_mean_rank = {}
    demographic_to_ranks = {}

    # Get the mean rank for each demographic
    for key in data:
        if key == 'Real_DDx': continue
        print(f'Calculating mean rank for {key}')
        mean_ranks, raw_predictions = get_mean_rank(data[key], real_order)
        demographic_to_mean_rank[key] = mean_ranks
        demographic_to_ranks[key] = raw_predictions
    
    num_samples = len(data['Female_Caucasian'])

    # Turn into DF and rename the columns
    mean_rank_df = pd.DataFrame(demographic_to_mean_rank)
    mean_rank_df.columns = [x.replace('_', ' ').title() for x in mean_rank_df.columns]
    #graph_mean_rank(mean_rank_df, args.output_dir, args.case_num)
    #mean_rank_df.to_csv(f'{args.output_dir}/mean_rank_case_{args.case_num}.csv')

    ## Calculate multivariate regression
    raw_predictions = pd.DataFrame(demographic_to_ranks)
    #is_significant = multiwave_anova(raw_predictions, num_samples)

    #raw_predictions.to_csv(f'{args.output_dir}/raw_predictions_case_{args.case_num}.csv')

    # Now we want to calculate the kallman tau rank statistic
    dx_to_kendall_tau = {}
    for key in data:
        if key == 'Real_DDx': continue
        kt = calculate_kendalltau(data[key], real_order)
        dx_to_kendall_tau[key] = kt

    # Now, create a violin plot
    kendall_tau_df = pd.DataFrame(dx_to_kendall_tau)
    p_value = kendall_tau_anova(kendall_tau_df)


    kendall_tau_df.to_csv(f'{args.output_dir}/kendall_tau_case_{args.case_num}.csv')
    graph_kendalltaus(kendall_tau_df, args.output_dir, args.case_num, args.topic, p_value)

    # # Get the Kendall Tau t-test + anova results
    # sex_wise_ttest_on_kendall_tau(kendall_tau_df)
    # race_wise_chitest_on_kendall_tau(kendall_tau_df)
    # all_chitest_kendall_tau(kendall_tau_df)
    
