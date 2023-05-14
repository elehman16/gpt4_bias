import os
import json
import argparse
import pandas as pd
import seaborn as sns
from typing import Tuple
from collections import Counter
import matplotlib.pyplot as plt

def parse_with_options(s: str, options: list[str]) -> str:
    """Match to the best fitting option. """
    matched = None
    for o in options:
        if o in s and matched is None:
            matched = o
        
        elif o in s and matched is not None:
            return 'Cannot parse'

    return matched if matched is not None else 'Cannot parse'

def parse_statements(statements: list[str], options: list[str]) -> Counter[str]:
    """Parse the statements and return a counter of the responses. """
    statements = statements[0].values

    all_responses = []
    for s in statements:

        if '\n\n' in s:
            all_responses.append(parse_with_options(s.split('\n\n')[1], options))
        elif '\n' in s:
            all_responses.append(parse_with_options(s.split('\n')[-1], options))
        else: 
            all_responses.append(parse_with_options(s, options))

    # Only return the ones we could parse.
    counts = Counter(all_responses)
    del counts['Cannot parse']
    return counts

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
    return df, dropped_df


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_file_df', type=str, required=True)
    argparser.add_argument('--input_file_json', type=str, required=True)
    argparser.add_argument('--output_dir', type=str, required=True)
    args = argparser.parse_args()

    # Load the data
    df, dropped_df = load_data(args.input_file_df, args.input_file_json)

    # Now, we want to parse the statements and get the counts for each statement.
    case_to_statement_to_performance, statement_to_options = {}, {}
    for demographics, group in dropped_df.groupby(['statement', 'race', 'gender', 'case']):
        all_statements = parse_statements(group.response.values, eval(group.options.iloc[0]))
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

    # Check that we can save into the output data directory.
    os.makedirs(args.output_dir, exist_ok=True)

    # Now, we have a dictionary of case to statement to performance. We can now compute the bias.
    # For each case, we want to compute the bias for each statement.
    # For each statement, we want to compute the bias for each demographic.
    # We have to take into account that we need to normalize by the number of samples.
    for q in case_to_statement_to_performance:
        for s in case_to_statement_to_performance[q]:
            # Plot the bias for each demographic

            options = statement_to_options[s]
            dist_, demographics, all_options = [], [], []
            for d in case_to_statement_to_performance[q][s]:
                # First, let's get the total number of samples
                total_samples = sum(case_to_statement_to_performance[q][s][d].values())

                # Now, let's get the number of samples for each option 
                normalized_counts_per_option = [case_to_statement_to_performance[q][s][d].get(o, 0) / total_samples for o in options]

                # Add to dist_per_demographic
                dist_.extend(normalized_counts_per_option)
                demographics.extend([d] * len(options))
                all_options.extend(options)

            # Now do the plot! Should be a bar graph where the hue is the demographic.
            tmp_df = pd.DataFrame({'demographic': demographics, 'option': all_options, 'normalized_count': dist_})
            df_pivot = tmp_df.pivot(index='option', columns='demographic', values='normalized_count')
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

            plt.savefig(f"{args.output_dir}/{q}_{s}.pdf", bbox_inches='tight')
