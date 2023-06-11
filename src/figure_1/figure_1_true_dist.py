import pandas as pd

df_reg = pd.read_csv('True Prevalence of Diseases and Conditions - Eric.csv')

normalized = df_reg[df_reg['Type'] == 'Normalized']
unnormalized = df_reg[df_reg['Type'] == 'Unnormalized']

# Normalize the unnormalized data
TRUE_POPULATION_STATS = {
    'white': 57.8 / 100,
    'hispanic': 18.7 / 100,
    'black': 12.1 / 100,
    'asian': 5.9 / 100,
    'male': 49.1 / 100,
    'female': 50.9 / 100     
}

def normalize_percent(row: pd.Series) -> dict[str, float]:
    # Condition                        Hep B
    # Male                               5.3
    # Female                             3.4
    # Black/African American            10.8
    # White                              2.1
    # Hispanic/Latino                    3.8
    # Asian                             21.1
    # Total                              4.3
    # Units                                %
    p_condition = float(row['Total'])
    p_white = float(row['White']) 
    p_black = float(row['Black/African American'])
    p_hispanic = float(row['Hispanic/Latino'])
    p_asian = float(row['Asian'])
    p_male = float(row['Male'])
    p_female = float(row['Female'])

    # perform bayes theorem: p(A|B) = p(B|A) * p(A) / p(B)
    # p(race | condition) = p(condition | race) * p(race) / p(condition)
    # p(sex | condition) = p(condition | sex) * p (sex) / p(condition)
    
    # Calculate using bayes
    p_white_given_condition = p_white * TRUE_POPULATION_STATS['white'] / p_condition
    p_black_given_condition = p_black * TRUE_POPULATION_STATS['black'] / p_condition 
    p_hispanic_given_condition = p_hispanic * TRUE_POPULATION_STATS['hispanic'] / p_condition
    p_asian_given_condition = p_asian * TRUE_POPULATION_STATS['asian'] / p_condition
    
    # Calculate using bayes
    p_condition_given_male = p_male * TRUE_POPULATION_STATS['male'] / p_condition
    p_condition_given_female = p_female * TRUE_POPULATION_STATS['female'] / p_condition

    # Sanity check
    assert p_condition_given_female + p_condition_given_male > 0.97
    total_p = p_white_given_condition + p_black_given_condition + p_hispanic_given_condition + p_asian_given_condition
    if total_p < 0.925:
        print(f"Warning! Total probability is less than 92.5% for condition {row.Condition}: {total_p}")

    return {
        'Condition': row.Condition,
        'Male': p_condition_given_male * 100,
        'Female': p_condition_given_female * 100,
        'Black/African American': p_black_given_condition * 100,
        'White': p_white_given_condition * 100,
        'Hispanic/Latino': p_hispanic_given_condition * 100,
        'Asian': p_asian_given_condition * 100,
        'Missing Race': max(0, 1 - total_p) * 100,
        'Missing Sex': max(1 - (p_condition_given_female + p_condition_given_male), 0) * 100
    }


def normalize_non_percent(row: pd.Series):
    units = int(''.join(row['Units'].split(',')))
    # Male                                     83.0
    # Female                                   27.0
    # Black/African American                     53
    # White                                      53
    # Hispanic/Latino                            66
    # Asian                                      46
    # Total                                    57.0
    # Units                                 100,000

    p_condition = float(row['Total']) / units
    p_white = float(row['White']) / units
    p_black = float(row['Black/African American']) / units
    p_hispanic = float(row['Hispanic/Latino']) / units
    p_asian = float(row['Asian']) / units
    p_male = float(row['Male']) / units
    p_female = float(row['Female']) / units

    # perform bayes theorem: p(A|B) = p(B|A) * p(A) / p(B)
    # p(race | condition) = p(condition | race) * p(race) / p(condition)
    # p(sex | condition) = p(condition | sex) * p (sex) / p(condition)
    
    # Calculate using bayes
    p_white_given_condition = p_white * TRUE_POPULATION_STATS['white'] / p_condition
    p_black_given_condition = p_black * TRUE_POPULATION_STATS['black'] / p_condition 
    p_hispanic_given_condition = p_hispanic * TRUE_POPULATION_STATS['hispanic'] / p_condition
    p_asian_given_condition = p_asian * TRUE_POPULATION_STATS['asian'] / p_condition
    
    # Calculate using bayes
    p_condition_given_male = p_male * TRUE_POPULATION_STATS['male'] / p_condition
    p_condition_given_female = p_female * TRUE_POPULATION_STATS['female'] / p_condition

    # Sanity check
    assert p_condition_given_female + p_condition_given_male > 0.97
    total_p = p_white_given_condition + p_black_given_condition + p_hispanic_given_condition + p_asian_given_condition
    if total_p < 0.925:
        print(f"Warning! Total probability is less than 92.5% for condition {row.Condition}: {total_p}")

    return {
        'Condition': row.Condition,
        'Male': p_condition_given_male * 100,
        'Female': p_condition_given_female * 100,
        'Black/African American': p_black_given_condition * 100,
        'White': p_white_given_condition * 100,
        'Hispanic/Latino': p_hispanic_given_condition * 100,
        'Asian': p_asian_given_condition * 100,
        'Missing Race': max(0, 1 - total_p) * 100,
        'Missing Sex': (1 - (p_condition_given_female + p_condition_given_male)) * 100
    }

ans1 = normalize_non_percent(unnormalized.iloc[0])
ans2 = normalize_percent(unnormalized.iloc[1])

# 'Male', 'Female', 'Black/African American', 'White', 'Hispanic/Latino', 'Asian'
df = pd.DataFrame([ans1, ans2])

normalized = normalized[['Condition', 'Male', 'Female', 'Black/African American', 'White', 'Hispanic/Latino', 'Asian']]
normalized = normalized.replace('-', 0.0)
# Cast columns as floats
normalized['Male'] = normalized['Male'].astype(float)
normalized['Female'] = normalized['Female'].astype(float)
normalized['Black/African American'] = normalized['Black/African American'].astype(float)
normalized['White'] = normalized['White'].astype(float)
normalized['Hispanic/Latino'] = normalized['Hispanic/Latino'].astype(float)
normalized['Asian'] = normalized['Asian'].astype(float)

missing_race_percentages = []
missing_sex_percentages = []

# Iterate over each row in the DataFrame
for index, row in normalized.iterrows():
    # Calculate count for 'Missing Race' and 'Missing Sex'
    missing_race_count = 100 - row[['Black/African American', 'White', 'Hispanic/Latino', 'Asian']].sum()
    missing_sex_count = 100 - row[['Male', 'Female']].sum()

    # Append the percentages to the respective lists
    missing_race_percentages.append(missing_race_count)
    missing_sex_percentages.append(missing_sex_count)

# Add the percentages as new columns in the DataFrame
normalized['Missing Race'] = missing_race_percentages
normalized['Missing Sex'] = missing_sex_percentages

# Concatenate the unnormalized and normalized DataFrames
final_ = pd.concat([df, normalized])
final_.to_csv('normalized_pt1.csv', index=False)