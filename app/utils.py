from dotenv import load_dotenv
from genai.credentials import Credentials
from genai.model import Model
from genai.schemas import GenerateParams
import json
import os
import re

import logging

load_dotenv()

GENAI_KEY = os.environ["GENAI_KEY"]
GENAI_API = os.environ["GENAI_API"]

def generate_prompt(
    instruction: str,
    functions: bool = False,
    functions_text: str = "",
    classes: bool = False,
    classes_text: str = "",
    documentation: bool = False,
    documentation_text: str = "",
    imports: bool = False,
    imports_text: str = "",
    other: bool = False,
    other_text: str = "",
) -> str:

    prompt = f"""{instruction}\n"""

    if functions:
        prompt += f"""functions:
```
{functions_text}
```
"""

    if classes:
        prompt += f"""classes:
```
{classes_text}
```
"""

    if documentation:
        prompt += f"""documentation:
```
{documentation_text}
```
"""

    if imports:
        prompt += f"""imports:
```
{imports_text}
```
"""

    if other:
        prompt += f"""other:
```
{other_text}
```
"""

    return prompt


def get_data() -> dict[str, list[dict]]:
    with open("data/raw/nested_dict.json", "r") as infile:
        code = json.load(infile)

    return code

def check_prompt_token_limit(
    model_id: str,
    prompt: str
) -> str:
    """
    Check if a given prompt is within the token limit of a model.

    Args:
        model_id (str): The model ID.
        prompt (str): The text prompt to check.

    Returns:
        str: A message indicating if the prompt is within or over the token limit.
    """

    # Initialize credentials and model
    creds = Credentials(GENAI_KEY, api_endpoint=GENAI_API)
    model = Model(model_id, credentials=creds)

    # Get the model card and token limit
    model_card = model.info()
    token_limit = int(model_card.token_limit)

    # Tokenize the prompt and count tokens
    tokenized_response = model.tokenize([prompt], return_tokens=True)
    prompt_tokens = int(tokenized_response[0].token_count)
    
    diff = prompt_tokens - token_limit

    # Check if prompt is within or over the token limit
    if token_limit >= prompt_tokens:
        return True, diff
    else:
        return False, diff

def generate_text(
    model_id: str, prompt: str, decoding_method: str, max_new_tokens: int, temperature: float, top_k: int, top_p: float
):
    creds = Credentials(GENAI_KEY, api_endpoint=GENAI_API)

    # Instantiate parameters for text generation
    params = GenerateParams(
        decoding_method=decoding_method,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Instantiate a model proxy object to send your requests
    model = Model(model_id, credentials=creds, params=params)

    response = model.generate([prompt])
    generated_patch = response[0].generated_text

    return generated_patch