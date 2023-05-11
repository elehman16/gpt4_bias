import json
import os
import openai
import asyncio
from typing import Any, Dict, List, Optional

from tap import Tap

# Arguments class
class Arguments(Tap):
    """Arguments for running the script."""
    
    # Model parameters
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 256
    top_p: float = 1.0
    num_samples: int = 50
    batch_size: int = 64

    # Input and output files
    # N.B. Either input_file (prompt) or input_data must be provided
    input_file: Optional[str] = '/Users/machine/Research/gpt4_bias/data/sarcoidosis/promptv2.txt' # .txt file with prompt
    input_data: Optional[str] = None # JSON string with prompt
    output_file: str = "output/sarcoidosis/demographics_case_report_promptv2_temp07.json"


async def dispatch_openai_requests(
    messages_list: List[List[Dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> List[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.

    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

# Main function
def main(args: Arguments) -> None:
    """Main function for running the script."""
    # Read input file
    if args.input_file is not None:
        # Open .txt file and read prompt
        with open(args.input_file, "r") as f:
            prompt = f.read().strip()
            inputs = [prompt] * args.num_samples
            prompt = ''
    elif args.input_data is not None:
        # Open the .json file and read all data
        with open(args.input_data, "r") as f:
            data = json.load(f)
            prompt = data["prompt"]
            inputs = data["inputs"]
    else:
        raise ValueError("Either input_file or input_data must be provided.")
    
    # If the output directory does not exist, create it
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))

    # Create messages list
    message_list = [
        [{"role": "user", 
          "content": f"{prompt} {input}" if prompt is not None else input}]
          for input in inputs
    ]

    predictions = []
    # Send them in batches
    BATCH_SIZE = args.batch_size
    for i in range(0, len(message_list), BATCH_SIZE):
        print(f"Sending messages {i} to {i+BATCH_SIZE}")
        batch = message_list[i:i+BATCH_SIZE]
        predictions_batch = asyncio.run(
            dispatch_openai_requests(
                messages_list=batch,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
            )
        )
        predictions.extend(predictions_batch)

        # Save the intermediate results to file (in case of crash)
        RESULTS = {
        "inputs": message_list,
        "predictions": predictions,
        }

        # Save results to file
        with open(args.output_file, "w") as f:
            json.dump(RESULTS, f, indent=4)

    for i, x in enumerate(predictions):
        print(f"Response {i}: {x['choices'][0]['message']['content']}\n\n")

    

if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args)


