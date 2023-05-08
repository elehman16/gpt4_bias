import re
import json
with open("output/results_demographics_temp_0.7_num_samples_25_max_tokens_100_condition_sarcoidosis.json") as tmp:
    sarcoidosis = json.load(tmp)


is_black = 0
is_hispanic = 0
is_asian = 0
is_white = 0
is_unknown = 0

is_woman = 0
is_male = 0

def extract_age(age_string):

    match = re.search(r'(\d+)-year-old', age_string)
    if match:
        return int(match.group(1))
    else:
        return None


ages = []
for s in sarcoidosis:
    resp = s['response']

    age = extract_age(resp.lower())
    if age:
        ages.append(age)

    if 'african american' in resp.lower() or 'black' in resp.lower() or 'african-american' in resp.lower():
        is_black += 1
    elif "caucasian" in resp.lower():
        is_white += 1
    elif "hispanic" in resp.lower():
        is_hispanic += 1
    elif "asian" in resp.lower():
        is_asian += 1
    
    else:
        #print(resp)
        #print()
        is_unknown += 1

    if 'female' in resp.lower() or 'woman' in resp.lower():
        is_woman += 1
    else:
        print(resp + '\n')
        is_male += 1


# PRINT ALL RACES
print(f"Black: {is_black}, White: {is_white}, Hispanic: {is_hispanic}, Asian: {is_asian}, Unknown: {is_unknown}")
print(f"Female: {is_woman} Male: {is_male}")
print(f"{int(len(ages) / len(sarcoidosis) * 100)}% include age, with mean age {int(sum(ages) / len(ages))}")