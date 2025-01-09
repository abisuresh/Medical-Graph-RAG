Fork of Medical-Graph-RAG
Conda environment requirements:

make sure to follow below instructions
please make sure Neo4j GDS library is installed locally pip install graphdatascience
it is helpful to have Neo4j desktop installed
in addition, make sure the following packages are installed in the environment:
pydantic 2.9.2 [can check version with this line python -c "import pydantic; print(pydantic.__version__)"]
anthropic 0.42.0
openai 1.58.1
requests 2.32.3
numpy
colorama 0.4.6
unstructured 0.16.11
langchain 0.3.13
langchain-community 0.3.13
neo4j 5.27.0
tiktoken 0.8.0
networkx 3.4.2
pymilvus 2.5.2

When passing in input, create a directory called dataset_ex and then add in a text file such as the example below called report_0 with arrays of input.

Get Neo4j database running locally if not using a cloud provider by installing Neo4j Desktop. AuraDB seems to have difficulty with the neo4j graph data science plugin which is required for this project.

Neo4j desktop install: https://neo4j.com/download/?utm_source=GSearch&utm_medium=PaidSearch&utm_campaign=Evergreen&utm_content=AMS-Search-SEMCE-DSA-None-SEM-SEM-NonABM&utm_term=&utm_adgroup=DSA&gad_source=1&gclid=Cj0KCQiA4fi7BhC5ARIsAEV1YiZODS9KEEMlA1VOvy7LKhkKnm_XGz8wF-RdV1k2uWXQQj0TE_NNmCwaAn2OEALw_wcB

make sure to create a database through the Neo4j desktop and set the details of username, password and uri as environment variables [NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_URI] on the terminal in your project directory and environment as mentioned in step 2 below
make sure that apoc and gds are installed in that database (using the neo4j desktop)
Key Debugging Note: If you are on a Mac, please run this command in the project directory before running the command to create the graph find . -name '.DS_Store' -type f -delete [i.e. before running step 3 below]

In order to add in prompts to this system, add in a text file called prompt.txt to the root of the directory. Add in questions in the following format to the prompt:

["What is the main symptom of the patient?"], ["The main symptom of the patient is severe chest pain"]

# Medical-Graph-RAG
We build a Graph RAG System specifically for the medical domain.

Check our paper here: https://arxiv.org/abs/2408.04187

## Quick Start (Baseline: a simple Graph RAG pipeline on medical data)
1. conda env create -f medgraphrag.yml

2. export OPENAI_API_KEY = your OPENAI_API_KEY

3. python run.py -simple True (now using ./dataset_ex/report_0.txt as RAG doc, "What is the main symptom of the patient?" as the prompt, change the prompt in run.py as you like.)

## Build from scratch (Complete Graph RAG flow in the paper)

### About the dataset
#### Paper Datasets
**Top-level Private data (user-provided)**: we used [MIMIC IV dataset](https://physionet.org/content/mimiciv/3.0/) as the private data.

**Medium-level Books and Papers**: We used MedC-K as the medium-level data. The dataset sources from [S2ORC](https://github.com/allenai/s2orc). Only those papers with PubMed IDs are deemed as medical-related and used during pretraining. The book is listed in this repo as [MedicalBook.xlsx](https://github.com/MedicineToken/Medical-Graph-RAG/blob/main/MedicalBook.xlsx), due to licenses, we cannot release raw content. For reproducing, pls buy and process the books.

**Bottom-level Dictionary data**: We used [Unified Medical Language System (UMLS)](https://www.nlm.nih.gov/research/umls/index.html) as the bottom level data. To access it, you'll need to create an account and apply for usage. It is free and approval is typically fast.

In the code, we use the 'trinity' argument to enable the hierarchy graph linking function. If set to True, you must also provide a 'gid' (graph ID) to specify which graphs the top-level should link to. UMLS is largely structured as a graph, so minimal effort is required to construct it. However, MedC-K must be constructed as graph data. There are several methods you can use, such as the approach we used to process the top-level in this repo (open-source LLMs are recommended to keep costs down), or you can opt for non-learning-based graph construction algorithms (faster, cheaper, and generally noisier)

#### Example Datasets
Recognizing that accessing and processing all the data mentioned may be challenging, we are working to provide simpler example dataset to demonstrate functionality. Currently, we are using the mimic_ex [here](https://huggingface.co/datasets/Morson/mimic_ex) here as the Top-level data, which is the processed smaller dataset derived from MIMIC. For Medium-level and Bottom-level data, we are in the process of identifying suitable alternatives to simplify the implementation, welcome for any recommendations.

### 1. Prepare the environment, Neo4j and LLM
1. conda env create -f medgraphrag.yml


2. prepare neo4j and LLM (using ChatGPT here for an example), you need to export:

export OPENAI_API_KEY = your OPENAI_API_KEY

export NEO4J_URL= your NEO4J_URL

export NEO4J_USERNAME= your NEO4J_USERNAME

export NEO4J_PASSWORD= your NEO4J_PASSWORD

### 2. Construct the graph (use "mimic_ex" dataset as an example)
1. Download mimic_ex [here](https://huggingface.co/datasets/Morson/mimic_ex), put that under your data path, like ./dataset/mimic_ex

2. python run.py -dataset mimic_ex -data_path ./dataset/mimic_ex(where you put the dataset) -grained_chunk -ingraphmerge -construct_graph

### 3. Model Inference
1. put your prompt to ./prompt.txt

2. python run.py -dataset mimic_ex -data_path ./dataset/mimic_ex(where you put the dataset) -inference

## Acknowledgement
We are building on [CAMEL](https://github.com/camel-ai/camel), an awesome framework for construcing multi-agent pipeline.

## Cite
~~~
@article{wu2024medical,
  title={Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation},
  author={Wu, Junde and Zhu, Jiayuan and Qi, Yunli},
  journal={arXiv preprint arXiv:2408.04187},
  year={2024}
}
~~~
