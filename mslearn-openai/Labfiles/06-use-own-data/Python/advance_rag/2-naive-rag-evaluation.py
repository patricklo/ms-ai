import os
from dotenv import load_dotenv, find_dotenv

def get_openai_api_key():
    _ = load_dotenv(find_dotenv())
    return os.getenv("OPENAI_API_KEY")

def get_openai_base_url():
    _ = load_dotenv(find_dotenv())
    return os.getenv("OPENAI_BASE_URL")

def get_hf_api_key():
    _ = load_dotenv(find_dotenv())
    return os.getenv("HUGGINGFACE_API_KEY")

# assign corresponding value to api_key before invoking OpenAI(), once setup here, all the following calling from
# other frameworks like LlamaIndex and Trulens will inherit and don't need to config for the same
import openai
openai.api_key = get_openai_api_key()
openai.base_url = get_openai_base_url()
#OPENAI_API_KEY = get_openai_api_key()
#OPENAI_BASE_URL = get_openai_base_url()

import numpy as np

import nest_asyncio
nest_asyncio.apply()

import warnings
warnings.filterwarnings('ignore')

"""## Naive RAG - Setup"""

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./eBook-How_to_Build_Your_Career_in_AI.pdf"]
).load_data()

#print(type(documents), "\n")
#print(len(documents), "\n")
#print(type(documents[0]), "\n")
#print(documents[0])

from llama_index.core import Document

document = Document(text="\n\n".join([doc.text for doc in documents]))

from llama_index.core import VectorStoreIndex
from llama_index.core import ServiceContext
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
)

# customize to use embedding model from HuggingFace
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
)

# when using from_documents, the document will be splitted into chunks and parsed into node objects, which store in memory
# by default, and VectorStoreIndex deals with vectors in batches of 2048 nodes, thus if the memory is constrained, we can
# modify insert_batch_size
# ref: https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index/
index = VectorStoreIndex.from_documents(
    [document],
    service_context=service_context,
    #insert_batch_size=512,
)

query_engine = index.as_query_engine()

response = query_engine.query(
    "What are steps to take when finding projects to build your experience?"
)
print(str(response))

"""## Naive RAG - Evaluation"""

eval_questions = []
with open('eval_questions_lite.txt', 'r') as file:
    for line in file:
        # remove newline character and convert to integer
        item = line.strip()

        #print(item)
        eval_questions.append(item)

from trulens_eval import Tru

tru = Tru()
tru.reset_database()

from trulens_eval.feedback.provider.openai import OpenAI
openai = OpenAI()

from trulens_eval import (
    Feedback,
    TruLlama,
    OpenAI,
)

qa_relevance = (
    Feedback(
        openai.relevance_with_cot_reasons,
        name="Answer Relevance",
    )
    # the input is the prompt of user and the output is the generative answer of LLM
    .on_input_output()
)

# the contexts are those retrieved thru RAG
context_selection = TruLlama.select_source_nodes().node.text

qs_relevance = (
    Feedback(
        openai.relevance_with_cot_reasons,
        name="Context Relevance",
    )
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)

#from trulens_eval.feedback import Groundedness
#grounded = Groundedness(groundedness_provider=openai)
#grounded = Groundedness(groundedness_provider=openai, summarize_provider=openai)

groundedness = (
    Feedback(openai.groundedness_measure_with_cot_reasons, name = "Groundedness")
    .on(context_selection)
    .on_output()
    #.aggregate(grounded.grounded_statements_aggregator)
)

feedbacks = [qa_relevance, qs_relevance]

def get_trulens_recorder(query_engine, feedbacks, app_id):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks,
    )
    return tru_recorder

def get_prebuilt_trulens_recorder(query_engine, app_id):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks,
    )
    return tru_recorder

tru_recorder = get_prebuilt_trulens_recorder(
    query_engine,
    app_id="Direct Query Engine",
)

with tru_recorder as recording:
    for question in eval_questions:
        response = query_engine.query(question)

records, feedback = tru.get_records_and_feedback(app_ids=[])
records.head()

# dashboard will be launched by localtunnel service on google-colab
#tru.run_dashboard()

