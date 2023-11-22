from utils import check_prompt_token_limit, generate_text, generate_prompt
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
        "verify_verifier"
    ],
)

logging.debug("user selected datapoint")

# load nested data
dataset_path = "../data/raw/chunked_data.json"
with open(dataset_path, 'r') as f:
		data = json.load(f)

logging.debug("loaded data")

code = data[file]["code_chunks"]

actual_doc = data[file]["markdown"][0]

with st.sidebar:
    st.header("Model Parameters")
    model_id = st.selectbox(
        label="Model",
        options=[
            "codellama/codellama-34b-instruct",
            "ibm/granite-20b-code-instruct-v1",
            "meta-llama/llama-2-13b",
            "ibm/granite-3b-code-plus-v1",
            "meta-llama/llama-2-70b"
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
        """
        Create documentation for the function below

        For Example:

        Function:
        def in_validity_period(self) -> bool:
                ###
                Returns whether or not this `Identity` is currently within its self-stated validity period.

                NOTE: As noted in `Identity.__init__`, this is not a verifying wrapper;
                the check here only asserts whether the *unverified* identity's claims
                are within their validity period.
                ###

                now = datetime.now(timezone.utc).timestamp()

                if self._nbf is not None:
                    return self._nbf <= now < self._exp
                else:
                    return now < self._exp

        Documentation:
        Returns whether or not this Identity is currently within its self-stated validity period.
        NOTE: As noted in Identity.__init__, this is not a verifying wrapper; the check here only asserts whether the         unverified identity's claims are within their validity period.",
            )
        """
    )

    st.write("Prompt Elements")
    functions = st.toggle("Functions", value=False)
    classes = st.toggle("Classes", value=False)
    documentation = st.toggle("Documentation", value=False)
    imports = st.toggle("Imports", value=False)
    other = st.toggle("Other", value=False)


functions_text = code["functions"]
# classes_text = code["classes"]
documentation_text = code["documentation"]
imports_text = code["imports"]
other_text = code["other"]

prompt = generate_prompt(
    instruction,
    functions=functions,
    functions_text=functions_text,
    # classes=classes,
    # classes_text=classes_text,
    documentation=documentation,
    documentation_text=documentation_text,
    imports=imports,
    imports_text=imports_text,
    other=other,
    other_text=other_text
)

print(functions)
print(prompt)

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