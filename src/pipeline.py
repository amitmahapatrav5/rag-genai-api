import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


# Get Context
embedding_model = HuggingFaceEmbeddings(
        model_name=os.environ.get('EMBEDDING_MODEL_REPO_ID')
)

db = Chroma(
    persist_directory=Path(os.environ.get('VECTOR_DB_PERSISTENT_ABSOLUTE_PATH')),
    embedding_function=embedding_model,
    collection_name=os.environ.get('VECTOR_DB_NAME')
)

retriever = db.as_retriever(
    search_type='similarity',
    k=5
)


# Prepare Prompt Template
prompt_template = PromptTemplate(
    template="""
        You are a helpful assistant that answers strictly based on the provided context.

        Guidelines:
        - Use only the information in the Context to answer the Query.
        - If the answer is not present in the Context, say "I don't have enough information in the provided context to answer that."
        - Be concise: 1 short paragraph, maximum 2-3 sentences.
        - Do not invent facts, do not speculate, and do not use external knowledge.
        - If multiple relevant points exist in Context, synthesize them clearly.
        - Preserve any important terminology from the Context.

        Context:
        {context}

        Query:
        {query}

        Answer:
    """,
    input_variables=["context", "query"],
)


llm = HuggingFaceEndpoint(
    repo_id=os.environ.get('CHAT_MODEL_REPO_ID'),
    task='text-generation'
)
chat_model = ChatHuggingFace(llm=llm)

output_parser = StrOutputParser()


# Build Pipeline
prompt_creation_chain = RunnableParallel(
    {
        'context': retriever | RunnableLambda(lambda docs: '\n\n'.join( [ doc.page_content for doc in docs ] )),
        'query': RunnablePassthrough()
    }
)

chain = prompt_creation_chain | prompt_template | chat_model | output_parser

# Created By Amit Mahapatra