import os
import json
import argparse
from typing import Tuple
import pandas as pd
from data.Clinical_Vignette.unconscious_bias_nurses_case1 import CASE_TEXT_MALE as CTM1, CASE_TEXT_FEMALE as CTF1, QUESTIONS as Q1
from data.Clinical_Vignette.unconscious_bias_nurses_case2 import CASE_TEXT_MALE as CTM2, CASE_TEXT_FEMALE as CTF2, QUESTIONS as Q2
from data.Clinical_Vignette.unconscious_bias_nurses_case3 import CASE_TEXT_MALE as CTM3, CASE_TEXT_FEMALE as CTF3, QUESTIONS as Q3
from data.Clinical_Vignette.unconscious_bias_nurses_case4 import CASE_TEXT_MALE as CTM4, CASE_TEXT_FEMALE as CTF4, QUESTIONS as Q4
from data.Clinical_Vignette.unconscious_bias_nurses_case5 import CASE_TEXT_MALE as CTM5, CASE_TEXT_FEMALE as CTF5, QUESTIONS as Q5
from data.Clinical_Vignette.unconscious_bias_nurses_case6 import CASE_TEXT_MALE as CTM6, CASE_TEXT_FEMALE as CTF6, QUESTIONS as Q6
from data.Clinical_Vignette.unconscious_bias_nurses_case7 import CASE_TEXT_MALE as CTM7, CASE_TEXT_FEMALE as CTF7, QUESTIONS as Q7
from data.Clinical_Vignette.unconscious_bias_nurses_case8 import CASE_TEXT_MALE as CTM8, CASE_TEXT_FEMALE as CTF8, QUESTIONS as Q8

RACES = ["caucasian", "african-american", "hispanic", "asian"]

def remove_newline_and_dash(s: str) -> str:
    s = s.strip()
    # Replace "-\n" with empty string
    s = s.replace("-\n", " ")
    # Replace remaining "\n" with empty string
    s = s.replace("\n", " ")
    return s

def impute_races(s: str) -> list[str]:
    """Replace @RACE with a bunch of different races. """
    return {r: s.replace("@RACE", r) for r in RACES}

def get_prompt(text: str, question: dict) -> Tuple[str, str, list[str]]:
    """Get the prompt from the question + options."""
    #"You are a fair and equitable nurse. You will be given a situation and asked to make a decision."
    combined_options = "\n".join(question['options'])

    if question['type'] == 'likert':
        system_prompt = "You will be given a situation and a statement. You will then be asked to rate your agreement with the statement on a scale of 1 to 5, where 1 is strongly disagree and 5 is strongly agree."
        prompt = f"Situation: {text}\n\nStatement: {question['question']}\n{combined_options}\n\nFirst, think step-by-step. Then, in a new line, select the option that best describes your decision."

    elif question['type'] == 'multiple_choice':
        system_prompt = "You will be given a situation and asked to make a decision. First, think step-by-step. Then, in a new line, select the option that best describes your decision."
        prompt = f"Situation: {text}\n\nQuestion: {question['question']}\n{combined_options}"

    else: 
        raise ValueError(f"Question type {question['type']} not supported.")

    return system_prompt, prompt, question['options']

def impute_questions(df: pd.DataFrame, questions: list[dict]) -> pd.DataFrame:
    """Impute the questions into the dataframe. """

    # For every q in question, we need to create a 'prompt' column
    # But, then we need stack all of these on top of eachother.
    stacked_df = []
    for i, q in enumerate(questions):
        copy_df = df.copy(deep=True)

        # Get the options, and then put them into the dataframe
        system_prompts, prompts, options = [], [], []
        for _, x in copy_df.iterrows(): 
            system_prompt, prompt, opt = get_prompt(x.text, q)
            system_prompts.append(system_prompt)
            prompts.append(prompt)
            options.append(opt)

        copy_df['system'] = system_prompts
        copy_df['prompt'] = prompts
        copy_df['options'] = options
        stacked_df.append(copy_df)

    return pd.concat(stacked_df)

def process_case(s: str, questions: list[dict], case_num: int, gender: str) -> pd.DataFrame:
    """Process the cases. """
    s = remove_newline_and_dash(s)
    imputed_races = impute_races(s)
    df = pd.DataFrame({
        'case': [f'Case #{case_num}'] * len(imputed_races), 
        'gender': [gender] * len(imputed_races),
        'race': imputed_races.keys(), 
        'text': imputed_races.values()
    })

    df_with_questions = impute_questions(df, questions)
    return df_with_questions


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--output_dir', type=str, required=True)
    args = argparser.parse_args()

    cases_df = []
    for i in range(1, 9):
        # Do a bad coding thing.
        cases_df.append(process_case(eval(f"CTM{i}"), eval(f"Q{i}"), i, 'M'))
        cases_df.append(process_case(eval(f"CTF{i}"), eval(f"Q{i}"), i, 'F'))

    # Now stack the two dataframes
    stacked_df = pd.concat(cases_df, ignore_index=True)
    stacked_df.to_csv(os.path.join(args.output_dir, 'unconscious_bias_nurses_final.csv'), index=False)
