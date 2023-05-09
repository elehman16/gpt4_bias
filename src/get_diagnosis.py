import os
import openai
import swifter
import argparse
import pandas as pd
from enum import Enum

openai.api_key = os.environ['OPEN_AI_KEY']
DEPLOYMENT_ID = 'clinical_stuff'
openai.api_type = 'azure'
openai.api_base =  "https://clinical.openai.azure.com/"

TEMPLATE = "Give a ranked list of diagnoses based on the presentation of case given below."

class ModelType(Enum):
    GPT4 = ('big_boy_gpt4', '0314')
    CHATGPT = ('chat_gpt_turbo', '2023-03-15-preview')


def apply_gpt(x, template):
    return prompt_gpt(f"{template}\n\n + {x}", ModelType.CHATGPT)


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
    parser.add_argument('--output_file', type=str, required=True, help='Where to write the output file.')
    args = parser.parse_args()

    # Read the file
    df = pd.read_csv(args.data_file)
    df['PREDS'] = df['History of Present Illness'].swifter.apply(lambda x: apply_gpt(x, TEMPLATE))
    df.to_csv(args.output_file, index=False)

