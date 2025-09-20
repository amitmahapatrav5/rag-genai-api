from typing import List, Dict, Optional, TextIO, BinaryIO, TypedDict
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_core.vectorstores import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TextSplitter


class PayloadType(TypedDict):
    file: TextIO | BinaryIO
    documents: List[Document]
    chunks: List[Document]


class Component(ABC):
    @abstractmethod
    def run(self, payload: Dict) -> Dict:
        pass


class Loader(Component):
    def __init__(self, loader: BaseLoader):
        self.loader = loader

    def run(self, payload: PayloadType):
        file: TextIO | BinaryIO = payload['file']
        loader = self.loader(file)
        payload['documents'] = loader.load()
        return payload


class Splitter(Component):
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


class Enricher(Component):
    # I want this metadata from user only, so has to get it via constructor
    def __init__(self, metadata: Dict[str, str]):
        self.metadata = metadata
    
    def run(self, payload: PayloadType):
        chunks: List[Document] = payload['chunks']
        for chunk in chunks:
            chunk.metadata.update(self.metadata.copy()) # important
        return payload


class VStore(Component):
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def run(self, payload: PayloadType):
        chunks: List[Document] = payload['chunks']
        self.vector_store.add_documents(chunks)
        return payload


# Created By Amit Mahapatra