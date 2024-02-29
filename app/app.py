from utils import (
    check_prompt_token_limit,
    generate_text,
    generate_prompt,
    generate_text_using_OpenAI,
    eval_using_model,
    indicate_key_presence,
    eval_using_langchain,
)
from feedback import store_feedback
import os
import streamlit as st
import logging
import json
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textstat import textstat
import os
from streamlit_feedback import streamlit_feedback


# Set theme, title, and icon
st.set_page_config(page_title="API Docs Generator", page_icon="📄", layout="wide")


def get_env_variable(var: str) -> str:
    env = os.getenv(var)
    if not env:
        raise ValueError(f"environment variable '{var}' is not set")
    return env


# Allow the user to provide their own API keys
user_genai_key = st.text_input(
    "Enter GENAI_KEY:", placeholder=indicate_key_presence("GENAI_KEY")
)
user_openai_key = st.text_input(
    "Enter OPENAI_API Key:", placeholder=indicate_key_presence("OPENAI_API_KEY")
)


# maybe it's a bit redundant to define these two functions but whatever
def GENAI_KEY() -> str:
    """
    Grabs the GENAI_KEY at the time that it's needed,
    either from the user input or from the environment
    """
    if user_genai_key:
        return user_genai_key.strip()
    return get_env_variable("GENAI_KEY")


def OPENAI_API_KEY() -> str:
    """
    Grabs the OPENAI_API_KEY at the time that it's needed,
    either from the user input or from the environment
    """
    if user_openai_key:
        return user_openai_key.strip()
    return get_env_variable("OPENAI_API_KEY")


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    handlers=[logging.FileHandler("/tmp/app.log"), logging.StreamHandler()],
)

logging.info("starting app")

st.title("API Docs Generator 📄", anchor="center")

logging.debug("loading data")


file = st.selectbox(
    "Select a file to work with",
    [
        "errors",
        "oidc",
        "sign",
        "transparency",
        "verify_models",
        "verify_policy",
        "verify_verifier",
    ],
)

logging.debug("user selected datapoint")

# load nested data
DATASET_PATH = os.getenv("DATASET_PATH", "data/raw/chunked_data.json")

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

logging.debug("loaded data")

code = data[file]["code_chunks"]

actual_doc = data[file]["markdown"]

with st.sidebar:
    st.header("Model Parameters")
    model_id = st.selectbox(
        label="Model",
        options=[
            "ibm/granite-20b-code-instruct-v1",
            "codellama/codellama-34b-instruct",
            "meta-llama/llama-2-13b",
            "ibm/granite-3b-code-plus-v1",
            "meta-llama/llama-2-70b",
            "OpenAI/gpt3.5",
            "bigcode/starcoder",
            "tiiuae/falcon-180b",
        ],
    )

    decoding_method = st.selectbox(
        label="Decoding Strategy", options=["sample", "greedy"]
    )

    max_new_tokens = st.slider(
        label="Max New Tokens", min_value=0, max_value=1024, value=300, step=1
    )

    temperature = st.slider(
        label="Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.01
    )

    top_k = st.slider(label="Top K", min_value=1, max_value=100, value=50, step=1)

    top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    st.header("Prompt Builder")

    instruction = st.text_area(
        "Instruction",
        """
Generate API documentation for Python code provided in the prompt. Ensure clarity, accuracy, and user-centricity.
If no code is provided, do not speculate or generate generic examples. Instead, leave this section blank or state "No code provided".

If Python code is provided:

1. Introduction: 
2. Class Documentation:
  - Document each class present in the code, including:
    - Class Name and Description
    - Class Attributes and Data types
    - Documentation for each method within the class, following the instructions below.
3. Function Documentation:
  - For each function in the code:
    - Function Description
    - Parameters, including names and data types.
    - Return values, including data types.
4. Error Handling:
Describe possible error responses and how they are handled in the code.
""",
    )

    st.write("Prompt Elements")
    functions = st.toggle("Functions", value=False)
    classes = st.toggle("Classes", value=False)
    documentation = st.toggle("Documentation", value=False)
    imports = st.toggle("Imports", value=False)
    other = st.toggle("Other", value=False)
    functions_code = st.toggle("Functions Code only", value=False)
    functions_doc = st.toggle("Functions Documentation only", value=False)
    classes_code = st.toggle("Classes Code only", value=False)
    classes_doc = st.toggle("Classes Documentation only", value=False)

print(code.keys())

functions_text = code["functions"]
classes_text = code["classes"]
documentation_text = code["documentation"]
imports_text = code["imports"]
other_text = code["other"]
functions_code_text = code["functions_code"]
functions_doc_text = code["functions_docstrings"]
classes_code_text = code["classes_code"]
classes_doc_text = code["classes_docstrings"]

prompt = generate_prompt(
    instruction,
    functions=functions,
    functions_text=functions_text,
    classes=classes,
    classes_text=classes_text,
    documentation=documentation,
    documentation_text=documentation_text,
    imports=imports,
    imports_text=imports_text,
    other=other,
    other_text=other_text,
    functions_code=functions_code,
    functions_code_text=functions_code_text,
    functions_doc=functions_doc,
    functions_doc_text=functions_doc_text,
    classes_code=classes_code,
    classes_code_text=classes_code_text,
    classes_doc=classes_doc,
    classes_doc_text=classes_doc_text,
)

with st.expander("Expand to view prompt"):
    st.text_area(label="prompt", value=prompt, height=600)


def main(prompt_success: bool, prompt_diff: int, actual_doc: str):
    if not prompt_success:
        st.write(f"Prompt is {prompt_diff} tokens too long, please shorten it")
        return

    # Generate text
    logging.info("requesting generation from model %s", model_id)

    if model_id == "OpenAI/gpt3.5":
        result = generate_text_using_OpenAI(prompt, OPENAI_API_KEY())

    else:
        result = generate_text(
            model_id,
            prompt,
            decoding_method,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            GENAI_KEY(),
        )
    col1, col2, col3 = st.columns([1.5, 1.5, 0.5])

    with col1:
        st.subheader(f"Generated API Doc")
        for line in result.split("\n"):
            st.markdown(
                f'<div style="color: black; font-size: small">{line}</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.subheader("Actual API Doc")
        for line in actual_doc.split("\n"):
            st.markdown(
                f'<div style="color: black; font-size: small">{line}</div>',
                unsafe_allow_html=True,
            )

    with col3:
        st.subheader("Evaluation Metrics")

        st.markdown(
            "**LLM based evaluation on logic, correctness and helpfulness:**",
            help="Use Langchain Criteria based Eval to evaluate on cutsom criteria (this list can be updated based on what we are looking to see from the generated docs). Note this is language mo0del based evaluation and not always a true indication of the quality of the output that is generatged."
        )

        lc_score = eval_using_langchain(result, prompt, actual_doc)
        st.markdown(
            f"Logical: {lc_score[0]['score']}",
            help="Checks if the output is logical. Binary integer 0 to 1, where 1 would mean that the output is logical and 0 means it is not",
        )
        
        st.markdown(
            f"Helpfulness: {lc_score[1]['score']}",
            help="Checks if the output helpful for the end user. Binary integer 0 to 1, where 1 would mean that the output is helpful and 0 means it is not",
        )

        st.markdown(
            f"Correctness: {lc_score[2]['score']}",
            help="Checks if the output correct. Binary integer 0 to 1, where 1 would mean that the output is correct and accurate and 0 means it is not"
        )

if st.button("Generate API Documentation"):
    if model_id != "OpenAI/gpt3.5":
        prompt_success, prompt_diff = check_prompt_token_limit(
            model_id, prompt, GENAI_KEY()
        )

        main(prompt_success, prompt_diff, actual_doc)
    else:
        main(True, True, actual_doc)

# generate the feedback section now
streamlit_feedback(
    feedback_type="thumbs",
    on_submit=store_feedback,
    optional_text_label="Please tell us how we could make this more useful",
    align="flex-start",
)
