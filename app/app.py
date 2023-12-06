from utils import check_prompt_token_limit, generate_text, generate_prompt, generate_text_using_OpenAI, eval_using_model
import os
import streamlit as st
import logging
import json
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from readability import Readability
import textstat


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
    layout="wide"
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
You are an AI system specialized at generating API documentation for the provided Python code. You will be provided functions, classes, or Python scripts. Your documentation should include:

1. Introduction: Briefly describe the purpose of the API and its intended use.   
2. Functions: Document each API function, including:
    - Description: Clearly explain what the endpoint or function does.
    - Parameters: List and describe each parameter, including data types and any constraints.
    - Return Values: Specify the data type and possible values returned.

3. Error Handling: Describe possible error responses and their meanings.

Make sure to follow this output structure to create API documentation that is clear, concise, accurate, and user-centric. Avoid speculative information and prioritize accuracy and completeness.

"""
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

def main(prompt_success, prompt_diff, actual_doc):
    if not prompt_success:
        st.write(f"Prompt is {prompt_diff} tokens too long, please shorten it")
        return

    # Generate text
    logging.info("requesting generation from model %s", model_id)

    if model_id =="OpenAI/gpt3.5":
        result = generate_text_using_OpenAI(prompt)
        
    else:
        result = generate_text(
        model_id, prompt, decoding_method, max_new_tokens, temperature, top_k, top_p
        )
    col1, col2, col3 = st.columns([1.5, 1.5, 0.5])
    
    with col1:
        st.subheader(f"Generated API Doc")
        for line in result.split("\n"):
            st.markdown(
            f'<div style="color: black; font-size: small">{line}</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("Actual API Doc")
        for line in actual_doc.split("\n"):
            st.markdown(
            f'<div style="color: black; font-size: small">{line}</div>', unsafe_allow_html=True)

    with col3:
        st.subheader("Evaluation Metrics")
        # rouge score addition
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(actual_doc, result)
        st.markdown(f"ROUGE-1 Score:{rouge_scores['rouge1'].fmeasure:.2f}", help="ROUGE-1 refers to the overlap of unigrams (each word) between the system and reference summaries")
        st.markdown(f"ROUGE-2 Score: {rouge_scores['rouge2'].fmeasure:.2f}", help="ROUGE-2 refers to the overlap of bigrams between the system and reference summaries")
        st.markdown(f"ROUGE-L Score: {rouge_scores['rougeL'].fmeasure:.2f}", help="Longest common subsequence problem takes into account sentence-level structure similarity naturally and identifies longest co-occurring in sequence n-grams automatically")

        # calc cosine similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([actual_doc, result])
        cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
        st.markdown(f"Cosine Similarity Score: {cosine_sim[0][0]:.2f}", help="0 cosine similarity means no similarity between generated and actual API documentation, 1 means they are same")
        st.markdown("###") # add a line break
        
        st.markdown("**GenAI evaluation scores:**", help="Use OpenAI GPT 3 to evaluate the result of the generated API doc")
        score = eval_using_model(result)
        st.write(score)
        
        # Readability Scores
        st.subheader("Readability Metrics")

        # Flesch Reading Ease
        flesch_reading_ease = textstat.flesch_reading_ease(result)
        st.markdown(f"Flesch Reading Ease: {flesch_reading_ease:.2f}", help="Flesch Reading Ease measures how easy a text is to read. Higher scores indicate easier readability. Ranges 0-100")

        # Dale Chall Readability
        dale_chall_readability = textstat.dale_chall_readability_score(result)
        st.markdown(f"Dale Chall Readability: {dale_chall_readability:.2f}", help="The Dale-Chall Formula is an accurate readability formula for the simple reason that it is based on the use of familiar words, rather than syllable or letter counts. Lower scores mean more difficult words. No fixed ranges.")

        # Automated Readability Index (ARI)
        ari = textstat.automated_readability_index(result)
        st.markdown(f"ARI (Automated Readability Index): {ari:.2f}", help="ARI relies on a factor of characters per word, instead of the usual syllables per word. ARI corresponds to a U.S. grade level. Higher scores indicate more advanced reading levels.")


if st.button("Generate API Documentation"):
    
    if model_id != "OpenAI/gpt3.5":
        prompt_success, prompt_diff = check_prompt_token_limit(model_id, prompt)

        main(prompt_success, prompt_diff, actual_doc)
    else:
        
        main(True, True, actual_doc)
        