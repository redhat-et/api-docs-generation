from dotenv import load_dotenv
from genai.credentials import Credentials
from genai.model import Model
from genai.schemas import GenerateParams
import json
import os
import re
from openai import OpenAI

import logging

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
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
    functions_code: bool = False,
    functions_code_text: str = "",
    functions_doc: bool = False,
    functions_doc_text: str = "",
    classes_code: bool = False,
    classes_code_text: str = "",
    classes_doc: bool = False,
    classes_doc_text: str = "",
) -> str:

    functions_text_joined = '\n'.join(functions_text)
    classes_text_joined = '\n'.join(classes_text)
    documentation_text_joined = '\n'.join(documentation_text)
    imports_text_joined = '\n'.join(imports_text)
    other_text_joined = '\n'.join(other_text)
    functions_code_text_joined = '\n'.join(functions_code_text)
    functions_doc_text_joined = '\n'.join(functions_doc_text)
    classes_code_text_joined = '\n'.join(classes_code_text)
    classes_doc_text_joined= '\n'.join( classes_doc_text)
    # print(functions_text_joined)

    prompt = f"""{instruction}\n"""

    if functions and functions_text_joined:
        prompt += f"""

Function:

{functions_text_joined}

"""

    if functions_code and functions_code_text_joined:
        prompt += f"""


Function Code:

{functions_code_text_joined}

"""

    if functions_doc and functions_doc_text_joined:
        prompt += f"""


Function Docstrings:

{functions_doc_text_joined}

Documentation:

"""

    if classes and classes_text_joined:
        prompt += f"""
        
Class:

{classes_text_joined}

"""
    if classes_code and classes_code_text_joined:
        prompt += f"""
        
Class code:

{classes_code_text_joined}

""" 
    if classes_doc and classes_doc_text_joined:
        prompt += f"""

Classes Docstrings:

{classes_doc_text_joined}

Documentation
"""

    if documentation and documentation_text_joined:
        prompt += f"""
Here is some code documentation for reference:

{documentation_text_joined}

"""

    if imports and imports_text_joined:
        prompt += f"""
Here are the import statements for reference:

{imports_text_joined}

"""

    if other and other_text:
        prompt += f"""
Here are other lines of code for reference:

{other_text_joined}

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

    print(response)

    generated_patch = response[0].generated_text

    return generated_patch

def generate_text_using_OpenAI(prompt: str):
    
    creds = (OPENAI_API_KEY)
    client = OpenAI()
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": f"{prompt}"},
      ]
    )
    
    response = completion.choices[0].message.content
    print(response)
    return response


def eval_using_model(result):
    
    prompt = f"""Below is an API documentation for code, rate the documentation on factors such as Accuracy, Relevance,  Clarity, Completeness and Readability. Rate it on a scale of 1 to 5. 1 for the poorest documentation and 5 for the best.
    
    Example: 
    
    Accuracy: 1 
    Relevance: 2 
    Clarity: 3
    Completeness: 4 
    Readability: 5
    Overall Score: 3 
    
    Documentation:
    
    {result}
    
    GenAI Score: """
    response = generate_text_using_OpenAI(prompt)
    return response
    
    
    

