import pandas as pd

df_cond = pd.read_csv('True Prevalence of Diseases and Conditions - Eric -- conditional probability.csv')

# Normalize the unnormalized data
TRUE_POPULATION_STATS = {
    'white': 57.8 / 100,
    'hispanic': 18.7 / 100,
    'black': 12.1 / 100,
    'asian': 5.9 / 100,
    'male': 49.1 / 100,
    'female': 50.9 / 100     
}

df_cond.replace('-', 0, inplace=True)

df_cond['White Female'] = df_cond['White Female'].astype(float)
df_cond['White Male'] = df_cond['White Male'].astype(float)
df_cond['AA Female'] = df_cond['AA Female'].astype(float)
df_cond['AA Male'] = df_cond['AA Male'].astype(float)
df_cond['Latino Female'] = df_cond['Latino Female'].astype(float)
df_cond['Latino Male'] = df_cond['Latino Male'].astype(float)
df_cond['Asian Female'] = df_cond['Asian Female'].astype(float)
df_cond['Asian Male'] = df_cond['Asian Male'].astype(float)

normalized = df_cond[df_cond['Type'] == 'Normalized']
unnormalized = df_cond[df_cond['Type'] == 'Unnormalized']

def normalize_percent(row: pd.Series) -> dict[str, float]:
    p_condition = float(row['P(Condition)'].split('%')[0])
    p_white_female = row['White Female'] 
    p_white_male = row['White Male'] 
    p_aa_male = row['AA Male'] 
    p_aa_female = row['AA Female'] 
    p_hispanic_male = row['Latino Male'] 
    p_hispanic_female = row['Latino Female'] 
    p_asian_male = row['Asian Male'] 
    p_asian_female = row['Asian Female'] 
    
    # perform bayes theorem: p(A|B) = p(B|A) * p(A) / p(B)
    # p(race | condition) = p(condition | race) * p(race) / p(condition)
    # p(sex | condition) = p(condition | sex) * p (sex) / p(condition)
    
    # Calculate using bayes
    p_white_female_given_condition = p_white_female * TRUE_POPULATION_STATS['white'] * TRUE_POPULATION_STATS['female'] / p_condition
    p_white_male_given_condition = p_white_male * TRUE_POPULATION_STATS['white']  * TRUE_POPULATION_STATS['male'] / p_condition

    p_black_female_given_condition = p_aa_female * TRUE_POPULATION_STATS['black'] * TRUE_POPULATION_STATS['female']  / p_condition
    p_black_male_given_condition = p_aa_male * TRUE_POPULATION_STATS['black'] * TRUE_POPULATION_STATS['male'] / p_condition

    p_hispanic_male_given_condition = p_hispanic_male * TRUE_POPULATION_STATS['hispanic'] * TRUE_POPULATION_STATS['male']  / p_condition
    p_hispanic_female_given_condition = p_hispanic_female * TRUE_POPULATION_STATS['hispanic'] * TRUE_POPULATION_STATS['female']  / p_condition

    p_asian_male_given_condition = p_asian_male * TRUE_POPULATION_STATS['hispanic'] * TRUE_POPULATION_STATS['male']  / p_condition
    p_asian_female_given_condition = p_asian_female * TRUE_POPULATION_STATS['hispanic'] * TRUE_POPULATION_STATS['female'] / p_condition

    # Aggregate, ignoring distributions across sex.
    p_white_given_condition = p_white_female_given_condition + p_white_male_given_condition
    p_black_given_condition = p_black_female_given_condition + p_black_male_given_condition
    p_hispanic_given_condition = p_hispanic_male_given_condition + p_hispanic_female_given_condition
    p_asian_given_condition = p_asian_male_given_condition + p_asian_female_given_condition

    p_condition_given_female_tmp = p_white_female_given_condition + p_black_female_given_condition + p_hispanic_female_given_condition + p_asian_female_given_condition
    p_condition_given_male_tmp = p_white_male_given_condition + p_black_male_given_condition + p_hispanic_male_given_condition + p_asian_male_given_condition

    p_condition_given_female = p_condition_given_female_tmp / (p_condition_given_male_tmp + p_condition_given_female_tmp)
    p_condition_given_male = p_condition_given_male_tmp / (p_condition_given_male_tmp + p_condition_given_female_tmp)

    # Sanity check
    if not p_condition_given_female + p_condition_given_male > 0.97:
        import pdb; pdb.set_trace()

    total_p = int(p_white_given_condition * 100) + int(p_black_given_condition * 100) + int(p_hispanic_given_condition * 100) + int(p_asian_given_condition * 100)

    if total_p > 100:
        p_white_given_condition = p_white_given_condition / total_p
        p_black_given_condition = p_black_given_condition / total_p
        p_hispanic_given_condition = p_hispanic_given_condition / total_p
        p_asian_given_condition = p_asian_given_condition / total_p

    if total_p < 92.5:
        print(f"Warning! Total probability is less than 92.5% for condition {row['Condition / Disease']}: {total_p}")

    return {
        'Condition': row['Condition / Disease'],
        'Male': p_condition_given_male * 100,
        'Female': p_condition_given_female * 100,
        'Black/African American': p_black_given_condition * 100,
        'White': p_white_given_condition * 100,
        'Hispanic/Latino': p_hispanic_given_condition * 100,
        'Asian': p_asian_given_condition * 100,
        'Missing Race': max(0, 1 - total_p),
        'Missing Sex': (1 - (p_condition_given_female + p_condition_given_male)) * 100
    }


def normalize_non_percent(row: pd.Series):
    units = int(''.join(row['Unit'].split(',')))

    p_condition = float(row['P(Condition)'].split('%')[0]) / 100
    p_white_female = row['White Female'] / units
    p_white_male = row['White Male'] / units
    p_aa_male = row['AA Male'] / units
    p_aa_female = row['AA Female'] / units
    p_hispanic_male = row['Latino Male'] / units
    p_hispanic_female = row['Latino Female'] / units
    p_asian_male = row['Asian Male'] / units
    p_asian_female = row['Asian Female'] / units
    
    # Calculate using bayes
    # Aggregate, ignoring distributions across sex.
    p_white_given_condition = (p_white_female + p_white_male) * TRUE_POPULATION_STATS['white'] / p_condition
    p_black_given_condition = (p_aa_female + p_aa_male) * TRUE_POPULATION_STATS['black'] / p_condition
    p_hispanic_given_condition = (p_hispanic_female + p_hispanic_male) * TRUE_POPULATION_STATS['hispanic'] / p_condition
    p_asian_given_condition = (p_asian_female + p_asian_male) * TRUE_POPULATION_STATS['asian'] / p_condition

    # One last normalization
    p_condition_given_female_tmp = (p_aa_female + p_white_female + p_asian_female + p_hispanic_female) * TRUE_POPULATION_STATS['female'] / p_condition
    p_condition_given_male_tmp = (p_aa_male + p_white_male + p_asian_male + p_hispanic_male) * TRUE_POPULATION_STATS['male'] / p_condition
    p_condition_given_female = p_condition_given_female_tmp / (p_condition_given_female_tmp + p_condition_given_male_tmp)
    p_condition_given_male = p_condition_given_male_tmp / (p_condition_given_female_tmp + p_condition_given_male_tmp)

    # Sanity check
    if not p_condition_given_female + p_condition_given_male > 0.97:
        import pdb; pdb.set_trace()

    total_p = int(p_white_given_condition * 100) + int(p_black_given_condition * 100) + int(p_hispanic_given_condition * 100) + int(p_asian_given_condition * 100)

    if total_p > 100:
        p_white_given_condition = p_white_given_condition / total_p
        p_black_given_condition = p_black_given_condition / total_p
        p_hispanic_given_condition = p_hispanic_given_condition / total_p
        p_asian_given_condition = p_asian_given_condition / total_p

    if total_p < 92.5:
        print(f"Warning! Total probability is less than 92.5% for condition {row['Condition / Disease']}: {total_p}")

    return {
        'Condition': row['Condition / Disease'],
        'Male': p_condition_given_male * 100,
        'Female': p_condition_given_female * 100,
        'Black/African American': p_black_given_condition * 100,
        'White': p_white_given_condition * 100,
        'Hispanic/Latino': p_hispanic_given_condition * 100,
        'Asian': p_asian_given_condition * 100,
        'Missing Race': max(0, 1 - total_p),
        'Missing Sex': (1 - (p_condition_given_female + p_condition_given_male)) * 100
    }

# Normalize the data
unnormalized_perc = unnormalized[unnormalized['Unit'] == '%']
unnormalized_units = unnormalized[unnormalized['Unit'] != '%']

normalized_rows = []
for i, row in unnormalized_perc.iterrows():
    normalized_rows.append(normalize_percent(row))

for i, row in unnormalized_units.iterrows():
    normalized_rows.append(normalize_non_percent(row))


final_normalized = pd.DataFrame(normalized_rows)

# Now convert the `normalized` dataframe to the rows we want.
converted_normalized = []
for i, row in normalized.iterrows():
    converted_normalized.append({
        'Condition': row['Condition / Disease'],
        'Male': row['AA Male'] + row['White Male'] + row['Latino Male'] + row['Asian Male'],
        'Female': row['AA Female'] + row['White Female'] + row['Latino Female'] + row['Asian Female'],
        'Black/African American': row['AA Male'] + row['AA Female'] ,
        'White': row['White Male'] + row['White Female'],
        'Hispanic/Latino': row['Latino Male'] + row['Latino Female'],
        'Asian': row['Asian Male'] + row['Asian Female'],
    })

converted_normalized = pd.DataFrame(converted_normalized)

missing_race_percentages = []
missing_sex_percentages = []

new_male, new_female = [], []
# Iterate over each row in the DataFrame
for index, row in converted_normalized.iterrows():
    # Calculate count for 'Missing Race' and 'Missing Sex'
    missing_race_count = 100 - row[['Black/African American', 'White', 'Hispanic/Latino', 'Asian']].sum()
    missing_sex_count = 0.0 #100 - row[['Male', 'Female']].sum()

    # Append the percentages to the respective lists
    missing_race_percentages.append(missing_race_count)
    missing_sex_percentages.append(missing_sex_count)
    new_male.append(row['Male'] / (row['Male'] + row['Female']))
    new_female.append(row['Female'] / (row['Male'] + row['Female']))


# Add the percentages as new columns in the DataFrame
converted_normalized['Male'] = new_male
converted_normalized['Female'] = new_female
converted_normalized['Missing Race'] = missing_race_percentages
converted_normalized['Missing Sex'] = missing_sex_percentages

# Concat 
df = pd.concat([final_normalized, converted_normalized])
df.to_csv('normalized_pt2.csv', index=False)
#normalize_percent(normalized.iloc[0])
#normalize_non_percent(normalized.iloc[0])
