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

"""
****************************************************************
******************************* Option1: Naive RAG - Setup
****************************************************************
"""

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

"""
****************************************************************
***************** Option2: Sentence-window Retrieval - Setup
****************************************************************
"""

from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core import load_index_from_storage

def build_sentence_window_index(
        document,
        llm,
        embed_model="local:BAAI/bge-small-en-v1.5",
        save_dir="sentence_index",
):
    # create the sentence window node parser with default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        # for each sentence, the parser will include 3 sentences before and after it respectively within the metadata
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )

    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            [document],
            service_context=sentence_context,
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index

def get_sentence_window_query_engine(
        sentence_index,
        similarity_top_k=6,
        rerank_top_n=2,
):
    # define post-processor to replace with sentence-window
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")

    # ref: https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/SentenceTransformerRerank/
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n,
        model="BAAI/bge-reranker-base",
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=[postproc, rerank],
    )

    return sentence_window_engine

sentence_index = build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index",
)

sentence_window_engine = get_sentence_window_query_engine(sentence_index)

window_response = sentence_window_engine.query(
    "How do I get started on a personal project in AI?"
)
print(str(window_response), "\n")

# compare the content from sentence-window with the original sentence
window = window_response.source_nodes[0].node.metadata["window"]
sentence = window_response.source_nodes[0].node.metadata["original_text"]

print(f"Original sentence:\n{sentence}\n")
print(f"Sentence-window content:\n{window}")

"""## Sentence-window Retrieval - Evaluation"""

tru.reset_database()

tru_recorder_sentence_window = get_prebuilt_trulens_recorder(
    sentence_window_engine,
    app_id = "Sentence-window Query Engine",
)

for question in eval_questions:
    with tru_recorder_sentence_window as recording:
        response = sentence_window_engine.query(question)
        print(question)
        print(str(response))

leaderboard = tru.get_leaderboard(app_ids=[])
leaderboard.head()
#tru.run_dashboard()

"""
****************************************************************
******************************* Option3: Auto-merging Retrieval - Setup
****************************************************************
"""

from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core import StorageContext
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine

def build_automerging_index(
        documents,
        llm,
        embed_model="local:BAAI/bge-small-en-v1.5",
        save_dir="merging_index",
        chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            service_context=merging_context,
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )

    return automerging_index

def get_automerging_query_engine(
        automerging_index,
        similarity_top_k=12,
        rerank_top_n=2,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)

    # ref: https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/
    retriever = AutoMergingRetriever(
        base_retriever,
        automerging_index.storage_context,
        verbose=True,
    )

    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n,
        model="BAAI/bge-reranker-base",
    )

    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=[rerank],
    )

    return auto_merging_engine

automerging_index = build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
)

automerging_query_engine = get_automerging_query_engine(
    automerging_index,
)

auto_merging_response = automerging_query_engine.query(
    "How do I build a portfolio of AI projects?"
)
print(str(auto_merging_response))

"""## Auto-merging Retrieval - Evaluation"""

tru.reset_database()

tru_recorder_automerging = get_prebuilt_trulens_recorder(
    automerging_query_engine,
    app_id="Auto-merging Query Engine",
)

for question in eval_questions:
    with tru_recorder_automerging as recording:
        response = automerging_query_engine.query(question)
        print(question)
        print(str(response))

leaderboard = tru.get_leaderboard(app_ids=[])
leaderboard.head()
#tru.run_dashboard()

