import os
import openai
from tqdm import tqdm
import concurrent.futures
import time

openai.api_key = os.environ["OPENAI_API_KEY"]

MODEL='gpt-4'
def query_openai(query: list, temperature: float = 0.7, max_tokens: int = 100) -> str:
    """Query OpenAI with the given a of prompt."""
    output = openai.ChatCompletion.create(
        model=MODEL,
        messages=query,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return output["choices"][0]["message"]["content"]

def query_openai_single_sample(prompt: list[dict], temperature, max_tokens):
    response = query_openai(prompt, temperature=temperature, max_tokens=max_tokens)
    time.sleep(1)
    return {
        'prompt': prompt,
        'response': response
    }

def run_prompts(prompts: list[list[dict]], num_samples: int, temperature: float, max_tokens: int, n_workers: int = 4) -> list[str]:
    """Run the given LIST of prompts and return the results."""
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_list = []
        for prompt in prompts:
            for n in range(num_samples):
                future = executor.submit(query_openai_single_sample, prompt, temperature, max_tokens)
                future_list.append(future)

        for future in tqdm(concurrent.futures.as_completed(future_list), total=len(future_list)):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error occurred: {e}")

    return results