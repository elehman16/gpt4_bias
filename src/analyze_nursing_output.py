import json
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def parse_with_options(s, options):
    matched = None
    for o in options:
        if o in s and matched is None:
            matched = o
        
        elif o in s and matched is not None:
            return 'Cannot parse'

    return matched if matched is not None else 'Cannot parse'

def parse_statements(statements: list[str], options: list[str]):
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

df = pd.read_csv('output/unconscious_bias_nurses_v2.csv')
jf_white = json.load(open("output/gpt_4_outputs_051323_nursing.json"))
jf_black = json.load(open("output/african_american_bias_nursing_gpt4.json"))

# Load black and white
prompts, answers = [], []
for row in jf_white:
    prompts.append(row['prompt'][1]['content'])
    answers.append(row['response'])

for row in jf_black:
    prompts.append(row['prompt'][1]['content'])
    answers.append(row['response'])

# Put it into a DF
df_prompt_answers = pd.DataFrame({'prompt': prompts, 'response': answers})

# Group by prompt

#aggr_ = df_prompt_answers.groupby('prompt').agg(list)
aggr = pd.DataFrame([{'prompt': key, 'response': x['response']} for key, x in df_prompt_answers.groupby('prompt')])

# Combine with df on the prompt column, and add a new column for the response
merged_df = pd.merge(df, aggr, on='prompt', how='left')
dropped_df = merged_df.dropna()

# Group by statement
dropped_df['statement'] = [x.split('\n\n')[1].split('\n')[0] for x in dropped_df.prompt]
#grouped_statements = {'statement': [], 'race': [], 'gender': [], 'response_dist': [], 'case': []}

case_to_statement_to_performance = {}
statement_to_options = {}
    
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

        plt.savefig(f"output/figures/{q}_{s}.pdf", bbox_inches='tight')


        #sns.barplot(data=pd.DataFrame(dist_per_demographic), x=options, y=list(dist_per_demographic.keys()), hue=list(dist_per_demographic.keys()))
