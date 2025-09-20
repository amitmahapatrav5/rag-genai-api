import os
from typing import TextIO, BinaryIO

from dotenv import load_dotenv
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


load_dotenv()

class InMemoryFileObjectLoader(BaseLoader):
    def __init__(self, file: TextIO|BinaryIO):
        self.file = file
        return self
    
    def load(self):
        return [Document(page_content=self.file.read())]



loader = InMemoryFileObjectLoader(file=file)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', ' ', ''], chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)


db = Chroma(
    persist_directory=os.environ.get('VECTOR_DB_PERSISTENT_ABSOLUTE_PATH'),
    collection_name='content',
)

# Created By Amit Mahapatra