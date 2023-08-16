import os
import json
import argparse
import pandas as pd
import seaborn as sns
from typing import Tuple
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.sandbox.stats.multicomp import multipletests

def run_per_race_lr(df: pd.DataFrame, dem1: str, dem2: str):
    # Only keep the race columns that have the race field

    df = df[df['demographics'].isin([dem1, dem2])]

    # Create a single dummy column
    dummy_column = pd.get_dummies(df['demographics'], columns=['demographics'], drop_first=True)
    dummy_column = dummy_column.astype(int)

    # Concatenate the dummy column with the original DataFrame
    df_encoded = pd.concat([df['answers'], dummy_column], axis=1)

    # Our target variable is 'answers'
    y = df['answers']

    # Remove 'answers' from the DataFrame to create our feature matrix
    X = df_encoded.drop(columns='answers')

    # Fit the model
    model = OrderedModel(y, X, distr='logit')
    result = model.fit(method='bfgs')
    pvals = result.pvalues.iloc[:1].values
    demographics = result.pvalues.iloc[:1].index.values
    _, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
    

    return print(f"Evaluation of: {dem1} vs. {dem2} -- {dummy_column.keys()[0]} is 1\n\n{pvals_corrected}\n\n{pvals}\n\n")

def run_ordinal_lr(df: pd.DataFrame):
    # We need to drop one otherwise we will have perfect multicollinearity
    # in our model. i.e., one variable can be directly predicted by inversing the other.
    df_encoded = pd.get_dummies(df, columns=['demographics'])
    del df_encoded['demographics_caucasian_M'] #df.drop(columns=['demographics_caucasian_M'])

    # Our target variable is 'answers'
    y = df['answers']

    # Remove 'answers' from the DataFrame to create our feature matrix
    X = df_encoded.drop(columns='answers')

    # Fit the model
    try:
        model = OrderedModel(y, X, distr='logit')
        result = model.fit(method='bfgs')
        pvals = result.pvalues.iloc[:7].values
        demographics = result.pvalues.iloc[:7].index.values
        _, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
        
    except: 
        default_dem = [
            'demographics_african-american_F', 
            'demographics_african-american_M',
            'demographics_asian_F', 
            'demographics_asian_M',
            'demographics_caucasian_F', 
            'demographics_hispanic_F',
            'demographics_hispanic_M'
        ]

        return {x: 1.0 for x in default_dem}

    return {demographic: pval for demographic, pval in zip(demographics, pvals_corrected)}


def parse_with_options(s: str, options: list[str]) -> str:
    """Match to the best fitting option. """
    matched = None
    for o in options:
        if o in s or o in s.replace(':', '.') and matched is None:
            matched = o
        
        elif o in s and matched is not None:
            return 'Cannot parse'
    
    if matched is None:
        return 'Cannot parse'

    return matched 

def parse_statements(statements: list[str], options: list[str]) -> Tuple[Counter[str], int]:
    """Parse the statements and return a counter of the responses. """
    statements = statements[0].values

    all_responses = []
    for s in statements:

        if '\n\n' in s:
            all_responses.append(parse_with_options(s.split('\n\n')[-1], options))
        elif '\n' in s:
            all_responses.append(parse_with_options(s.split('\n')[-1], options))
        else: 
            all_responses.append(parse_with_options(s, options))

    # Only return the ones we could parse.    
    counts = Counter(all_responses)
    num_cannot_parse = counts['Cannot parse']
    del counts['Cannot parse']
    return counts, num_cannot_parse

def load_data(df_path: str, json_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the data from the given paths and parse into a format that works best for us."""
    df = pd.read_csv(df_path)
    jf_data = json.load(open(json_path))

    # Load black and white
    prompts, answers = [], []
    for row in jf_data:
        prompts.append(row['prompt'][1]['content'])
        answers.append(row['response'])

    # Put it into a DF
    df_prompt_answers = pd.DataFrame({'prompt': prompts, 'response': answers})
    aggr = pd.DataFrame([{'prompt': key, 'response': x['response']} for key, x in df_prompt_answers.groupby('prompt')])

    # Combine with df on the prompt column, and add a new column for the response
    merged_df = pd.merge(df, aggr, on='prompt', how='left')
    dropped_df = merged_df.dropna()

    # Group by statement
    dropped_df['statement'] = [x.split('\n\n')[1].split('\n')[0] for x in dropped_df.prompt]
    df['statement'] = [x.split('\n\n')[1].split('\n')[0] for x in df.prompt]
    return df, dropped_df

def find_missing_samples(
    df: pd.DataFrame, 
    case_to_statement_to_performance: dict[str, dict[str, dict[str, dict[str, int]]]],
    all_samples: list[int], 
    ideal_sample: int
):
    # Now, we need to build a DF containing all samples that failed to run.
    # This means that we have to take the difference between the ideal sample and the actual sample.
    # If ideal_sample is -1, take the maximum in `all_samples`.
    # After, we will find the Race / Gender / Statement rows of the DF that don't meet our expectation.
    # We will duplicate them by the number of samples that we need to add.
    # We will then add them to a new DF and save it.
    if ideal_sample == -1:
        ideal_sample = max(all_samples)
    else:
        ideal_sample = ideal_sample

    # Find the missing samples
    missing_samples = []
    completley_missing_samples = []
    for index, row in df.iterrows():
        demographic_key = f"{row['race']}_{row['gender']}"
        statement = row['statement']
        case = row['case']

        if case in case_to_statement_to_performance and statement in case_to_statement_to_performance[case]:
            try: 
                current_sample_count = sum(case_to_statement_to_performance[case][statement][demographic_key].values())
            except KeyError:
                current_sample_count = 0

            missing_count = ideal_sample - current_sample_count
            if missing_count > 0:
                row_to_add = df[df.prompt == row.prompt].iloc[0]
                missing_samples.extend([row_to_add.to_dict()] * missing_count)
                print(f"Missing {missing_count} for {demographic_key} and {statement}")

        else:
            row_to_add = df[df.prompt == row.prompt].iloc[0]
            completley_missing_samples.extend([row_to_add.to_dict() for i in range(ideal_sample)])
            

    # Create a new dataframe with missing samples
    missing_samples_df = pd.DataFrame(missing_samples)

    # Save the missing samples dataframe to a CSV file
    missing_samples_df.to_csv(os.path.join(args.output_dir, "missing_samples.csv"), index=False)

# Sample Run
# PYTHONPATH=. python src/nursing_bias/analyze_nursing_output.py \
# --input_file_df ./output/nursing_bias/csv_outputs/unconscious_bias_nurses_final.csv \
# --input_file_json ./output/nursing_bias/json_outputs/final_outputs_0.7_25_samples_for_real.json \
# --pdf \ 
# --output_dir ./output/figures/nursing_bias/nursing_bias_25_samples_0.7_temp_v12.0/
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_file_df', type=str, required=True)
    argparser.add_argument('--ideal_sample', type=int, default=-1, help="How many samples were you TRYING to run with?")
    argparser.add_argument('--input_file_json', type=str, required=True)
    argparser.add_argument('--output_dir', type=str, required=True)
    argparser.add_argument('--pdf', action='store_true', help="Whether to save as a PDF or not.")
    args = argparser.parse_args()

    # Load the data
    df, dropped_df = load_data(args.input_file_df, args.input_file_json)

    # Now, we want to parse the statements and get the counts for each statement.
    case_to_statement_to_performance, statement_to_options = {}, {}
    total_num_not_parsed = 0
    for demographics, group in dropped_df.groupby(['statement', 'race', 'gender', 'case']):
        all_statements, num_not_parsed = parse_statements(group.response.values, eval(group.options.iloc[0]))
        case = demographics[3]

        # Set up the case to statement to performance
        if case_to_statement_to_performance.get(case) is None:
            case_to_statement_to_performance[case] = {}

        # Set up the statement to performance
        statement = demographics[0]
        if case_to_statement_to_performance[case].get(statement) is None:
            case_to_statement_to_performance[case][statement] = {}

        demographic_key = f"{demographics[1]}_{demographics[2]}"
        case_to_statement_to_performance[case][statement][demographic_key] = all_statements
        statement_to_options[statement] = eval(group.options.iloc[0])
        total_num_not_parsed += num_not_parsed

    print(f"Wasn't able to parse: {total_num_not_parsed}\n")

    # Check that we can save into the output data directory.
    os.makedirs(args.output_dir, exist_ok=True)

    # Now, we have a dictionary of case to statement to performance. We can now compute the bias.
    # For each case, we want to compute the bias for each statement.
    # For each statement, we want to compute the bias for each demographic.
    # We have to take into account that we need to normalize by the number of samples.
    all_samples = [] 
    p_values_map = []
    for q in case_to_statement_to_performance:
        for s in case_to_statement_to_performance[q]:
            # Plot the bias for each demographic

            options = statement_to_options[s]
            unnormalized_values, unnormalized_demographics = [], []
            dist_, demographics, all_options = [], [], []
            for d in case_to_statement_to_performance[q][s]:
                # First, let's get the total number of samples
                total_samples = sum(case_to_statement_to_performance[q][s][d].values())
                all_samples.append(total_samples)

                # Now, let's get the number of samples for each option 
                normalized_counts_per_option = [case_to_statement_to_performance[q][s][d].get(o, 0) / total_samples for o in options]

                # This is for the statistical significance test.
                for o in options:
                    samples = case_to_statement_to_performance[q][s][d].get(o, 0)
                    unnormalized_values.extend([o] * samples)
                    unnormalized_demographics.extend([d] * samples)

                # Add to dist_per_demographic
                dist_.extend(normalized_counts_per_option)
                demographics.extend([d] * len(options))
                all_options.extend(options)

            # Now do the plot! Should be a bar graph where the hue is the demographic.
            tmp_df = pd.DataFrame({'demographic': demographics, 'option': all_options, 'normalized_count': dist_})
            remap_demographics = {
                'african-american_F': 'Black Female',
                'african-american_M': 'Black Male',
                'asian_F': 'Asian Female',
                'asian_M': 'Asian Male',
                'hispanic_F': 'Hispanic Female',
                'hispanic_M': 'Hispanic Male',
                'caucasian_M': 'White Male',
                'caucasian_F': 'White Female'
            }

            tmp_df['demographic'] = tmp_df['demographic'].map(remap_demographics)
            tmp_df = tmp_df.sort_values('demographic')

            df_pivot = tmp_df.pivot(index='option', columns='demographic', values='normalized_count')
            if len(set(all_options)) == 5:
                answers_per_dm = pd.DataFrame({'demographics': unnormalized_demographics, 'answers': [int(o[0]) for o in unnormalized_values]})
                weighted_df = answers_per_dm.groupby('demographics').apply(lambda x: np.mean(x['answers'])).reset_index()
                weighted_df.columns = ['demographics', 'weighted_avg']

                std_df = answers_per_dm.groupby('demographics').apply(lambda x: np.std(x['answers'])).reset_index()
                std_df.columns = ['demographics', 'std']

                weighted_df.to_csv(f"{args.output_dir}/weighted_{q}_{s}.csv", index=False)
                std_df.to_csv(f"{args.output_dir}/std_{q}_{s}.csv", index=False)
            else:
                answers_per_dm = None

            # Do the test!
            ax = df_pivot.plot(kind='bar')

            # To set the labels of x-axis to an angle (e.g., 45 degrees)
            ax.set_xticklabels([])

            # Setting labels for the axes and the title of the plot
            plt.xlabel("")
            plt.ylabel("Proportion of Responses")
            plt.title("") #s)

            # To show the plot
            plt.tight_layout()
            sns.despine()
            plt.legend(loc=(1.04, 0))
            ax.get_legend().remove()

            if q == 'Case #1' and s == 'Statement: This patient is exaggerating their level of pain.':
                print("Case #1")
                df_pivot.to_csv(f"{args.output_dir}/case1_exaggerating.csv", index=False) 
                assert answers_per_dm is not None
                pvals_corrected = run_ordinal_lr(answers_per_dm)
            elif q == 'Case #2' and s == 'Statement: This patient is abusing Percocet.':
                print("Case #2")
                df_pivot.to_csv(f"{args.output_dir}/case2_abusing.csv", index=False)
                assert answers_per_dm is not None
                pvals_corrected = run_ordinal_lr(answers_per_dm)
            elif q == 'Case #6' and s == "Statement: This patient's family is hiding their alcohol abuse history.":
                print("Case #6")
                df_pivot.to_csv(f"{args.output_dir}/case6_hiding.csv", index=False)
                assert answers_per_dm is not None

                pvals_corrected = run_ordinal_lr(answers_per_dm)
                run_per_race_lr(answers_per_dm, 'hispanic_F', 'asian_F')
                run_per_race_lr(answers_per_dm, 'african-american_F', 'asian_F')
                import pdb; pdb.set_trace()

            elif answers_per_dm is not None:
                pvals_corrected = run_ordinal_lr(answers_per_dm)
                
            else:
                pvals_corrected = None

            if pvals_corrected is not None:
                p_values_map.append({'Statement': s, 'Case': q, 'p-values': pvals_corrected})

    # Now, we want to find the missing samples
    pvals = pd.DataFrame(p_values_map)

    to_cmb = pvals['p-values']
    pvals.drop(columns=['p-values'], inplace=True)

    demographics_to_perf = {}
    for x in to_cmb.values:
        for d in x.keys():
            if d not in demographics_to_perf:
                demographics_to_perf[d] = []

            demographics_to_perf[d].append(x[d])

    for key, value in demographics_to_perf.items():
        pvals[key] = value


    


