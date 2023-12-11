# Generating API docs using Generative AI methods

The objective of this project is to conduct a Proof of Concept (POC) exploring the feasibility of utilizing generative AI (GenAI) for writing API documentation. The primary focus involves taking the codebase for the Red Hat Trusted Artifact Signer (RHTAS) project and generating API documentation from it using generative AI. This documentation will then be compared against manually written human documentation to evaluate its overall quality.

## Links

* [Project Report](https://docs.google.com/document/d/1HYmC_LHrPTeyhSiCWqz0tGD-CRjjRrvmqvMYMSJjegU/edit?usp=sharingit status)
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

* Out of the models we tried, OpenAI GPT 3.5 generates the most readable and well-structured API documentation, it also parses code structures well and generates documentation for those.

* Among the rest of the models, llama-2-70b seems to also do well, it also roughly follows the expected API documentation structures and parses most simple code structures.

* Contrary to what we expected, code based models like IBM granite-20b-code-instruct and codellama-34b-instruct, are sometimes good at following the expected structure, but other times do not generate very readable documentation.


## Overall Viability 

* Generative AI models that we tried are performing sufficiently well and can be used for API documentation generation.

* These models can be integrated into a tool which can be used by developers and maintainers, streamlining the documentation process.

* Existing tools like GitHub CoPilot provide a mature interface and state-of-the-art results. In scenarios where there is a need to generate API docs for proprietary code, we see the use case for leveraging self hosted models.

* This can help with enhancing internal developer productivity, community building around our open source tooling. For open source code, leveraging state of the art models like GPT, Copilot seems reasonable.

