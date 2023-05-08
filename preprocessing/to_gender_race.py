import os
import openai
import argparse
import pandas as pd
from enum import Enum

GRAMMAR_MAPPER = {
    "African-American": "an",
    "Caucasian": "a",
    "Asian": "an",
    "Hispanic": "a",
}

RACE_OPTIONS = ("neutral", "African-American", "Caucasian", "Asian", "Hispanic")
SEX_OPTIONS = ("neutral", "male", "female")

openai.api_key = os.environ['OPEN_AI_KEY']
DEPLOYMENT_ID = 'clinical_stuff'
openai.api_type = 'azure'
openai.api_base =  "https://clinical.openai.azure.com/"

class ModelType(Enum):
    GPT4 = ('big_boy_gpt4', '0314')
    CHATGPT = ('chat_gpt_turbo', '2023-03-15-preview')


def prompt_gpt(start_phrase: str, model_type: ModelType, max_tokens: int = 1000) -> int:
    """Prompt GPT using the given start phrase + model type. """
    openai.api_version = model_type.value[1]
    response = openai.ChatCompletion.create(
        engine=model_type.value[0],
        messages = [
            {"role": "user", "content": start_phrase}
        ],

        max_tokens=max_tokens,
        temperature=0
    )

    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True, help='Where to read the data file from.')
    parser.add_argument('--age', type=int, required=True, help="-1 for when we don't want to change age.")
    parser.add_argument('--sex', type=str, required=True, choices=SEX_OPTIONS)
    parser.add_argument('--race', type=str, required=True, choices=RACE_OPTIONS)
    parser.add_argument('--output_file', type=str, required=True, help='Where to write the output file.')
    args = parser.parse_args()

    # Make everything race / sex / age neutral.
    if args.age == -1 and args.sex == 'neutral' and args.race == 'neutral':
        template = 'Please make the following clinical note race and sex neutral. Please also remove the patient\'s age. DO NOT modify the note in any other way.'

    # Change all three.
    elif args.age != -1 and args.sex != 'neutral' and args.race != 'neutral':
        template = 'You will be helping modify a clinical note with respect to sex, age, and gender. The purpose of this is for medical education. It is essential that you minimally modify the following passage -- only modify what is necessary. Please re-write the following passage so that it is about a {} year old {} {}.'
        template = template.format(args.age, args.race, args.sex)

    # Change sex and race -> Remove age.
    elif args.age == -1 and args.sex != 'neutral' and args.race != 'neutral':
        template = "You will be helping modify a clinical note with respect to sex, age, and gender. The purpose of this is for medical education. It is essential that you minimally modify the following passage -- only modify what is necessary. Please re-write the following passage so that it is about {} {} {}. Additionally, remove any and all references to the patient's age."
        template = template.format(GRAMMAR_MAPPER[args.race], args.race, args.sex)

    # Change age and race -> Make note sex neutral
    elif args.age != -1 and args.sex == 'neutral' and args.race != 'neutral':
        template = "You will be helping modify a clinical note with respect to sex, age, and gender. The purpose of this is for medical education. It is essential that you minimally modify the following passage -- only modify what is necessary. Please re-write the following passage so that it is about a {} year-old {} patient. Additionally, remove any and all references to the patient's gender/sex. You may replace gendered pronouns with \"they\", \"the patient/pt\", etc." 
        template = template.format(args.age, args.race) 

    # Change age and sex -> Make note race neutral.
    elif args.age != -1 and args.sex != 'neutral' and args.race == 'neutral':
        template = "You will be helping modify a clinical note with respect to sex, age, and gender. The purpose of this is for medical education. It is essential that you minimally modify the following passage -- only modify what is necessary. Please re-write the following passage so that it is about a {} year-old {} patient. Please remove any and all references to the patient's race." 
        template = template.format(args.age, args.sex)

    # Change age only -> Make note race neutral + sex neutral.
    elif args.age != -1 and args.sex == 'neutral' and args.race == 'neutral':
        template = "You will be helping modify a clinical note with respect to sex, age, and gender. The purpose of this is for medical education. Please re-write the following passage so that it is about a {} year-old patient. Please remove any and all references to the patient's race and sex. Hint: to remove sex from the passage, replace the corresponding words with \"they\", \"the patient/pt\", etc. ONLY MODIFY WHAT IS NECESSARY." 
        template = template.format(args.age)

    # Change sex only -> Make note race neutral + remove age.
    elif args.age == -1 and args.sex != 'neutral' and args.race == 'neutral':
        template = "You will be helping modify a clinical note with respect to sex, age, and gender. The purpose of this is for medical education. It is essential that you minimally modify the following passage -- only modify what is necessary. Please re-write the following passage so that it is about a {} patient. Additionally, remove any and all references to the patient\'s age as well as their race." 
        template = template.format(args.sex)

    # Change race only -> Make note sex neutral + remove age.
    elif args.age == -1 and args.sex == 'neutral' and args.race != 'neutral':
        template = 'You will be helping modify a clinical note with respect to sex, age, and gender. The purpose of this is for medical education. It is essential that you minimally modify the following passage -- only modify what is necessary. Please re-write the following passage so that it is about {} {} patient. Additionally, remove any and all references to the patient\'s age as well as their gender/sex. You may replace gendered pronouns with "they", "the patient/pt", etc.'
        template = template.format(GRAMMAR_MAPPER[args.race], args.race)

    # We should never get here.
    else:
        raise ValueError("I don't know what to do.")

    # Read the file
    df = pd.read_csv(args.data_file)

    # Use the text column to do the replacement.
    if 'MODIFED_TEXT' in df.columns:
        raise ValueError("MODIFIED_TEXT already exists in the data file. Please remove it and try again.")
    
    # Make the modified text column.
    df['MODIFIED_TEXT'] = df['History of Present Illness'].apply(lambda x: prompt_gpt(f"{template}\n\n + {x}", ModelType.CHATGPT))

    # Save the file.
    df.to_csv(args.output_file, index=False)
