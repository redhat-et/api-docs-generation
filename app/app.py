from utils import check_prompt_token_limit, generate_text
import os
import streamlit as st
import logging
import json

GENAI_KEY = os.environ["GENAI_KEY"]
GENAI_API = os.environ["GENAI_API"]

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    handlers=[logging.FileHandler("/tmp/app.log"), logging.StreamHandler()],
)

logging.info("starting app")

# Set theme, title, and icon
st.set_page_config(
    page_title="API Docs Generator",
    page_icon="📄",
)

st.title("API Docs Generator", anchor="center")

logging.debug("loading data")
# backports = load_data_with_defaults()
logging.debug("loaded data")


file = st.selectbox(
    "Select a file to work with",
    [
        "errors",
        "oidc",
        "sign",
        "transparency",
        "verify_models",
        "verify_policy",
        "verify_verifier"
    ],
)

logging.debug("user selected datapoint")

# load nested data
dataset_path = "../data/raw/nested_data.json"
with open(dataset_path, 'r') as f:
		data = json.load(f)
    
code = data[file]["code"][0]
actual_doc = data[file]["markdown"][0]

with st.sidebar:
    st.header("Model Parameters")
    model_id = st.selectbox(
        label="Model",
        options=[
            "codellama/codellama-34b-instruct",
            "ibm/granite-20b-code-instruct-v1",
            "meta-llama/llama-2-13b",
        ],
    )

    decoding_method = st.selectbox(
        label="Decoding Strategy", options=["greedy", "sample"]
    )

    max_new_tokens = st.slider(
        label="Max New Tokens", min_value=0, max_value=1024, value=1024, step=1
    )

    temperature = st.slider(
        label="Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.01
    )

    top_k = st.slider(label="Top K", min_value=1, max_value=100, value=50, step=1)

    top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    st.header("Prompt Builder")

    instruction = st.text_area(
        "Instruction",
        "Create API docs for the given code",
    )

    st.write("Prompt Elements")
    functions = st.toggle("Functions", value=False)
    classes = st.toggle("Classes", value=False)
    documentation = st.toggle("Documentation", value=False)
    imports = st.toggle("Imports", value=True)
    other = st.toggle("Other", value=True)


# functions = code["functions"]
# classes = code["classes"]
# documentation = code["documentation"]
# imports = code["imports"]
# other = code["other"]

# prompt = generate_prompt(
#     instruction,
#     functions=functions,
#     classes=classes,
#     documentation=documentation,
#     imports=imports,
#     other=other,
# )


prompt = """
Generate documentation for each function in the given code snippet:

{code}


""".format(code=code[f"{file}.py"])

with st.expander("Expand to view prompt"):
    st.text_area(label="prompt", value=prompt, height=600)

def main(prompt_success, prompt_diff):
    if not prompt_success:
        st.write(f"Prompt is {prompt_diff} tokens too long, please shorten it")
        return

    # Generate text
    logging.info("requesting generation from model %s", model_id)

    result = generate_text(
        model_id, prompt, decoding_method, max_new_tokens, temperature, top_k, top_p
    )

    st.text(result)


if st.button("Generate API Documentation"):
    prompt_success, prompt_diff = check_prompt_token_limit(model_id, prompt)

    main(prompt_success, prompt_diff)