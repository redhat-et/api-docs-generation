from genai import Credentials, Client
from genai.text.generation import TextGenerationParameters
from genai.text.tokenization import (
    TextTokenizationParameters,
    TextTokenizationReturnOptions,
    TextTokenizationCreateResults,
)
from langchain.evaluation import (
    Criteria,
    load_evaluator,
    EvaluatorType
)
import os
import json
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI


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
    functions_text_joined = "\n".join(functions_text)
    classes_text_joined = "\n".join(classes_text)
    documentation_text_joined = "\n".join(documentation_text)
    imports_text_joined = "\n".join(imports_text)
    other_text_joined = "\n".join(other_text)
    functions_code_text_joined = "\n".join(functions_code_text)
    functions_doc_text_joined = "\n".join(functions_doc_text)
    classes_code_text_joined = "\n".join(classes_code_text)
    classes_doc_text_joined = "\n".join(classes_doc_text)
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

Function Documentation:


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

Class Documentation:

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
    prompt: str,
    GENAI_KEY,
) -> (bool, str):
    """
    Check if a given prompt is within the token limit of a model.

    Args:
        model_id (str): The model ID.
        prompt (str): The text prompt to check.

    Returns:
        str: A message indicating if the prompt is within or over the token limit.
    """

    # Initialize credentials and model
    creds = Credentials(GENAI_KEY)
    client = Client(credentials=creds)

    # Get the model card and token limit
    model_details = client.model.retrieve(id=model_id)
    limits = model_details.result.token_limits
    token_limit = limits[0].token_limit

    # Tokenize the prompt and count tokens
    responses = list(
        client.text.tokenization.create(
            input=prompt,
            model_id=model_id,
            parameters=TextTokenizationParameters(
                return_options=TextTokenizationReturnOptions(tokens=True)
            ),
        )
    )
    results: TextTokenizationCreateResults = responses[0].results
    prompt_tokens = results[0].token_count
    diff = prompt_tokens - token_limit

    # Check if prompt is within or over the token limit
    return (token_limit >= prompt_tokens, diff)


def generate_text(
    model_id: str,
    prompt: str,
    decoding_method: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    genai_key: str,
):
    creds = Credentials(genai_key)

    # Instantiate parameters for text generation
    params = TextGenerationParameters(
        decoding_method=decoding_method,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Instantiate a model proxy object to send your requests
    client = Client(credentials=creds)
    responses = list(
        client.text.generation.create(
            model_id=model_id, inputs=[prompt], parameters=params
        )
    )
    response = responses[0].results[0]
    print(response)
    generated_patch = response.generated_text
    return generated_patch


def generate_text_using_OpenAI(prompt: str, openai_key: str):
    client = OpenAI(api_key=openai_key)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{prompt}"},
        ],
    )
    response = completion.choices[0].message.content
    print(response)
    return response


def eval_using_model(result: str, openai_key: str):
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
    response = generate_text_using_OpenAI(prompt, openai_key)
    return response


def indicate_key_presence(env: str) -> str:
    """
    This function will either return an empty string,
    or a string of '*' characters indicating the presence
    of said key.
    """
    key = os.getenv(env)
    if key:
        return "*" * len(key)
    else:
        return ""

def eval_using_langchain(prediction: str, query: str):

    evaluation = []
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    # If you wanted to specify multiple criteria. Generally not recommended
    custom_criterion_1 = {
        "grammatical": "Is the output grammatically correct?",
    }

    eval_chain = load_evaluator(EvaluatorType.CRITERIA, llm=llm, criteria=custom_criterion_1)

    eval_result = eval_chain.evaluate_strings(prediction=prediction, input=query)
    evaluation.append(eval_result)

    custom_criterion_2 = {
        "descriptive": "Does the output describe a piece of code and its intended functionality?"
    }

    eval_chain = load_evaluator(EvaluatorType.CRITERIA, llm=llm, criteria=custom_criterion_2)

    eval_result = eval_chain.evaluate_strings(prediction=prediction, input=query)
    evaluation.append(eval_result)

    evaluator = load_evaluator("criteria", llm=llm, criteria="helpfulness")

    eval_result = evaluator.evaluate_strings(prediction=prediction,input=query)
    evaluation.append(eval_result)

    return evaluation




