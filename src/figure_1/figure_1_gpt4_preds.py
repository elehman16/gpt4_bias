import re
import pandas as pd
import numpy as np
import pickle

dict_results = pickle.load(open('Dx_Dict_GPT4_x50 - New.pkl', 'rb'))
COND_NAMES = list(dict_results.keys())



def extract_age(input_text):
    match = re.search(r'(\d+)-year-old', input_text)
    if match:
        return int(match.group(1))
    else:
        match = re.search(r'(\d+) year-old', input_text)
        if match:
            return int(match.group(1))
        return None
    
def extract_race(input_text):
    races = ['White', 'Caucasian', 'Black', 'African American', 'African-American', 'Asian', 'Hispanic', 'Latino', 'Native American', 'American Indian', 'Alaskan Native', 'Native Hawaiian', 'Pacific Islander', 'Middle Eastern', 'Indian', 'Other']
    lowercase_races = [race.lower() for race in races]

    lower_text = input_text.lower()

    for i, race in enumerate(lowercase_races):
        if race in lower_text:
            return races[i]
    return None

def extract_gender(input_text):
    lower_text = input_text.lower()
    if 'female' in lower_text or 'woman' in lower_text:
        return 'Female'
    elif 'male' in lower_text or 'man' in lower_text:
        return 'Male'
    else:
        return None
    
CONDITIONS = {}
for cond_name in COND_NAMES:
    prompt_outputs = dict_results[cond_name]

    ages = []
    sexes = []
    ethnicities = []
    past_medical_histories = []

    for prompt_setting in prompt_outputs:
        for run in prompt_setting:
            output = run['response']
            try:
                age = output.split('Age:')[1].split('\n')[0].split(' years old')[0]
                sex = output.split('Sex:')[1].split('\n')[0]
                ethnicity = output.split('Ethnicity/Race:')[1].split('\n')[0]
                past_medical_history = output.split('Past Medical History:')[1].split('\n')[0]

            except:
                print(f'First error for {cond_name}')
                print('Trying again...')
                try:
                    age = str(extract_age(output))
                    sex = extract_gender(output)
                    ethnicity = extract_race(output)
                    past_medical_history = ''
                except:
                    print(f'Second error for {cond_name}')
                    print('***** Skipping...')

            age = age.strip()
            sex = sex.strip()
            ethnicity = ethnicity.strip()

            ages.append(age)
            sexes.append(sex)
            ethnicities.append(ethnicity)
            past_medical_histories.append(past_medical_history)

    CONDITIONS[cond_name] = {
        'ages': ages,
        'sexes': sexes,
        'race_ethnicities': ethnicities,
        'past_medical_histories': past_medical_histories,
    }

# Create a dataframe to store the results

dfs_pandas = []
for cond_name in COND_NAMES:
    df = pd.DataFrame(CONDITIONS[cond_name])
    df['condition'] = cond_name
    dfs_pandas.append(df)

df_all = pd.concat(dfs_pandas)

# Take a look at the distribution of race and ethnicity in the dataset
df_all['race_ethnicities'].value_counts().to_dict()


# // Adapted from Eric's code
# Concataenate all the race/ethnicity and gender results to plot them in a heatmap in seaborn
list_of_rates = []
for condition_name in COND_NAMES:
    df = pd.DataFrame(CONDITIONS[condition_name])

    SIZE = len(df)  # Number of samples
    
    ## GENDER
    gender = df['sexes'].value_counts().to_dict()
    is_male = 0
    is_female = 0
    is_gender_not_specified = 0
    
    # Add the "Male" and "Man" categories together
    for key in ["Male", "Man"]:
        is_male += gender.get(key, 0)

    # Add the "Female" and "Woman" categories together
    for key in ["Female", "Woman"]:
        is_female += gender.get(key, 0)

    # The rest is unspecified
    is_gender_not_specified = SIZE - is_male - is_female

    ## ETHNICITY
    ethnicity = df['race_ethnicities'].value_counts().to_dict()

    is_black = 0
    is_white = 0
    is_hispanic = 0
    is_asian = 0
    
    # Add the "Black" and "African American", "African-American" categories together to get the total number of Black patients in the dict
    # If the keys are not in the dict, add them with a value of 0
    for key in ['Black', 'African American', 'African-American']:
        is_black += ethnicity.get(key, 0)

    # Add the "White" and "Caucasian" categories together to get the total number of White patients in the dict
    for key in ['White', 'Caucasian']:
        is_white += ethnicity.get(key, 0)

    # Add the "Hispanic" and "Latino" categories together to get the total number of Hispanic patients in the dict
    for key in ['Hispanic', 'Latino']:
        is_hispanic += ethnicity.get(key, 0)

    # Add the "Asian" and "Asian American" categories together to get the total number of Asian patients in the dict
    for key in ['Asian', 'Asian American', 'Asian-American']:
        is_asian += ethnicity.get(key, 0)

    # The rest of the categories are unknown
    is_race_unknown = SIZE - is_black - is_white - is_hispanic - is_asian

    races = list(np.array([is_black,is_white,is_hispanic,is_asian,is_race_unknown])/SIZE)
    sexes = list(np.array([is_female,is_male ])/SIZE)
    print(f"Condition name: {condition_name}")
    print(f"Black: {is_black}, White: {is_white}, Hispanic: {is_hispanic}, Asian: {is_asian}, Unknown: {is_race_unknown}")
    print(f"Female: {is_female} Male: {is_male}")

    list_of_rates.append(races + sexes)
    
# Concatenate the list of rates into a numpy array
concatenated = np.array(list_of_rates)
row_labels = COND_NAMES
column_labels = ['Black', 'White', 'Hispanic','Asian','Other/Unknown','Female','Male']

tmp_df = pd.DataFrame(concatenated.tolist())
tmp_df.columns = column_labels
tmp_df.index = row_labels

tmp_df.to_csv('gpt4_demographics.csv', index=False)
