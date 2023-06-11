import os
import json
import argparse
import pandas as pd
import seaborn as sns
from typing import Tuple
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# For stats tests.
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg

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


    # Create a new dataframe with missing samples
    missing_samples_df = pd.DataFrame(missing_samples)

    # Save the missing samples dataframe to a CSV file
    missing_samples_df.to_csv(os.path.join(args.output_dir, "missing_samples.csv"), index=False)

def test_statistical_significance(df: pd.DataFrame, options: list[str]):
    """Do an ANOVA """
    # Assuming df is your DataFrame and 'group' is the column with your groups (White, Black, Asian)
    # and 'score' is the column with the response scores

    # Check normality for each group using Shapiro-Wilk test
    # Map answer to numbers
    mapping = {x: i for i, x in enumerate(options)}
    df['answers'] = df['answers'].map(mapping)

    # if df['answers'].isna().any():
    #     print("Non-numeric data found in 'answers' column. They were set to NaN.")
    #     raise ValueError("Error!")

    # for group in df['demographics'].unique():
    #     _, p = stats.shapiro(df[df['demographics'] == group]['answers'])
    #     if p < 0.05:
    #         print(f"The data for group {group} are not normally distributed.")
    #     else:
    #         print(f"The data for group {group} are normally distributed.")

    # # Check homogeneity of variance using Levene's test
    # _, p = stats.levene(*[df[df['demographics'] == group]['answers'] for group in df['demographics'].unique()])
    # if p < 0.05:
    #     print("The variances are not equal.")
    # else:
    #     print("The variances are equal.")

    # Perform ANOVA or Welch's ANOVA based on the variance check
    # if p < 0.05: # If variances are not equal
    #aov = pg.welch_anova(dv='answers', between='demographics', data=df)
    # else:
    model = ols('answers ~ C(demographics)', data=df).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    print(aov)
    print()

# Sample run: 
# PYTHONPATH=. python src/analyze_nursing_output.py \
#   --input_file_df output/nursing_bias/csv_outputs/unconscious_bias_nurses_final.csv \
#   --input_file_json output/nursing_bias/json_outputs/final_outputs_0.7_25_samples_for_real.json \
#   --output_dir output/figures/nursing_bias_25_samples_0.7_temp_4.0/ \
#   --pdf
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

            # Test the statistical significance.
            # answers_per_dm = pd.DataFrame({'demographics': unnormalized_demographics, 'answers': unnormalized_values})
            # if len(options) == 5:
            #     print(f"Statement: {s}")
            #     test_statistical_significance(answers_per_dm, options)
            # else:
            #     print("Skipping non-continuous options.\n\n\n")

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


            ax = df_pivot.plot(kind='bar')

            # To set the labels of x-axis to an angle (e.g., 45 degrees)
            plt.xticks(rotation=45, ha='right')

            # Setting labels for the axes and the title of the plot
            plt.xlabel("Option")
            plt.ylabel("Normalized Count")
            plt.title(s)

            # To show the plot
            plt.tight_layout()
            sns.despine()
            plt.legend(loc=(1.04, 0))
            ax.get_legend().remove()

            if args.pdf:
                plt.savefig(f"{args.output_dir}/{q}_{s}.pdf", bbox_inches='tight')
            else:
                plt.savefig(f"{args.output_dir}/{q}_{s}.png", dpi=400, bbox_inches='tight')

            plt.close()

    # Now, we want to find the missing samples
    #find_missing_samples(df, case_to_statement_to_performance, all_samples, args.ideal_sample)

# PYTHONPATH=. python src/analyze_nursing_output.py 
# --input_file_df ./output/nursing_bias/csv_outputs/unconscious_bias_nurses_final.csv 
# --input_file_json ./output/nursing_bias/json_outputs/final_outputs_0.7_25_samples_for_real.json 
# --pdf
# --output_dir ./output/figures/nursing_bias/nursing_bias_25_samples_0.7_temp_v6.0/
