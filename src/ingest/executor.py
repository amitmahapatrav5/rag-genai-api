from typing import List, Dict, Optional, TextIO, BinaryIO, TypedDict
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_core.vectorstores import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TextSplitter


class PayloadType(TypedDict):
    documents: List[Document]
    chunks: List[Document]


class Component(ABC):
    @abstractmethod
    def run(self, payload: Dict) -> Dict:
        pass


class LoaderExecutor(Component):
    def __init__(self, loader: BaseLoader):
        self.loader = loader

    def run(self, payload: PayloadType):
        payload['documents'] = self.loader.load()
        return payload


class SplitterExecutor(Component):
    def __init__(self, splitter: Optional[TextSplitter]):
        self.splitter = splitter or RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ' ', ''],
            chunk_size = 500,
            chunk_overlap = 100
        )
    
    def run(self, payload: PayloadType):
        documents = payload['documents']
        payload['chunks'] = self.splitter.split_documents(documents)
        return payload


class EnricherExecutor(Component):
    def __init__(self, metadata: Dict[str, str]):
        self.metadata = metadata
    
    def run(self, payload: PayloadType):
        chunks: List[Document] = payload['chunks']
        for chunk in chunks:
            chunk.metadata.update(self.metadata.copy()) # important
        return payload


class VStoreExecutor(Component):
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def run(self, payload: PayloadType):
        chunks: List[Document] = payload['chunks']
        self.vector_store.add_documents(chunks)
        return payload


# Created By Amit Mahapatra