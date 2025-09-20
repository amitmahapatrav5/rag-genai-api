from typing import TextIO, BinaryIO

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader


class InMemoryFileObjectLoader(BaseLoader):
    def __init__(self, file: TextIO|BinaryIO):
        self.file = file
    
    def load(self):
        return [Document(page_content=self.file.read())]