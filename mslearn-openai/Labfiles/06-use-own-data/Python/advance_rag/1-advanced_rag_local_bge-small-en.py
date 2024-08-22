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
