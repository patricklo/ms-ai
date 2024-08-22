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

"""## Auto-merging Retrieval - Setup"""

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

