# Generating API docs using Generative AI methods

The objective of this project is to conduct a Proof of Concept (POC) exploring the feasibility of utilizing generative AI (GenAI) for writing API documentation. The POC takes the codebase for the Red Hat Trusted Artifact Signer (RHTAS) project and generating API documentation from it using generative AI. This documentation is then evaluated in various ways for quality and correctness.

## Links

* [Project Report](https://docs.google.com/document/d/1HYmC_LHrPTeyhSiCWqz0tGD-CRjjRrvmqvMYMSJjegU/edit?usp=sharing)
* [Slides](https://docs.google.com/presentation/d/1xZ4729RXLi7FGjMAGuLzF8BUi5eH4qRDbmqs9eib30Q/edit?usp=sharing)
* [Planning and Design Doc](https://docs.google.com/document/d/1ToF-Z_XUAqUrpwHuCqFls85TwPYI7RT_vVFMuJDA7wU/edit?usp=sharing)


## Models tried

* GitHub Copilot
* GPT 3.5
* IBM Granite 20B
* IBM Granite 13B
* CodeLLAMA 34B
* LLAMA 70B
* starcoder
* falcon-180b

## Results

### Models
* Out of the models we tried, OpenAI GPT 3.5 generates the most readable and well-structured API documentation, it also parses code structures well and generates documentation for those. See comparison data against IBM granite in this [notebook](notebooks/evaluation/prompt_experiments.ipynb).

* Among the rest of the models, llama-2-70b seems to also do well, it also roughly follows the expected API documentation structures and parses most simple code structures.

* Contrary to what we expected, code based models like IBM granite-20b-code-instruct and codellama-34b-instruct, are sometimes good at following the expected structure, but other times do not generate very readable documentation.

### Evaluation
Automated evaluation of LLM generated text is a challenging task. We looked at methods like similairy scores (Cosine similarity, ROUGE scores) to compare the generated text against the actual human created documentation. We also looked at readability scores and grammatical checks to ensure that the text is readable and grammatically correct. Although none of the above scores give us a good undertsanding of the quality and usability of the generated output as well as the correctness and accuracy of these generated texts.

Thus we began to explore ways to use LLMs to evaluate LLM outputs. Upon experimentaion and comparing the results against human evaluation, we narrowed down on Langchain criteria based evaluation which uses OpenAI GPT in the background to score the generated text on helpfulness, correctness and logic (this list can be expanded based on requirements. The [study](notebooks/evaluation/quantitative_evaluation.ipynb) revealed these scores to be more effective than the other metrics we tried and to be consistent with human scoring majority of the times.

## Overall Viability 

* Generative AI models like GPT3.5 can read code and generate decently structured API documentation that follows a desired structure.

* These models could be integrated into a tool which can be used by developers and maintainers, streamlining the documentation process.

* Existing tools like GitHub CoPilot provide a mature interface and state-of-the-art results. In scenarios where there is a need to generate API docs for proprietary code, we see the use case for leveraging self hosted models.

* This can help with enhancing internal developer productivity, community building around our open source tooling. For open source code, leveraging state of the art models like GPT, Copilot seems reasonable.

