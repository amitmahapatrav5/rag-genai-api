import os

from langchain_core.document_loaders import BaseLoader
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter
)

from langchain_chroma import Chroma
from dotenv import load_dotenv

from ingest.cache import File


load_dotenv()


class LoaderSelector:
    def __init__(self, file: File):
        self.file = file
    
    def select(self) -> BaseLoader:
        if self.file.mimetype == '.txt':
            return TextLoader(file_path=self.file.filepath)
        elif self.file.mimetype == '.md':
            return UnstructuredMarkdownLoader(file_path=self.file.filepath)
        elif self.file.mimetype == '.pdf':
            return PyMuPDFLoader(file_path=self.file.filepath)
        elif self.file.mimetype == '.docx' or self.file.mimetype == '.docx':
            return UnstructuredWordDocumentLoader(file_path=self.file.filepath)
        elif self.file.mimetype == '.ppt' or self.file.mimetype == '.pptx':
            return UnstructuredPowerPointLoader(file_path=self.file.filepath)
        elif self.file.mimetype == '.csv':
            return CSVLoader(file_path=self.file.filepath)
        elif self.file.mimetype == '.xls' or self.file.mimetype == '.xlsx':
            return UnstructuredExcelLoader(file_path=self.file.filepath)

class SplitterSelector:
    def __init__(self, file: File, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.file = file
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def select(self, splitter_type: str = "default"):
        if splitter_type == "default":
            return self._get_default_splitter()
        
        # Specific splitter types
        splitter_map = {
            "recursive": self._get_recursive_splitter,
            "character": self._get_character_splitter,
            "token": self._get_token_splitter,
            "nltk": self._get_nltk_splitter,
            "spacy": self._get_spacy_splitter,
            "markdown_header": self._get_markdown_header_splitter,
            "markdown": self._get_markdown_splitter
        }
        
        if splitter_type in splitter_map:
            return splitter_map[splitter_type]()
        else:
            raise ValueError(f"Unknown splitter type: {splitter_type}")
    
    def _get_default_splitter(self):
        """Get the recommended default splitter for each file type"""
        if self.file.mimetype == '.txt':
            return self._get_recursive_splitter()
        elif self.file.mimetype == '.md':
            # return self._get_markdown_header_splitter()
            return self._get_markdown_splitter()
        elif self.file.mimetype in ['.pdf', '.docx', '.doc']:
            return self._get_recursive_splitter()
        elif self.file.mimetype in ['.ppt', '.pptx']:
            return self._get_recursive_splitter()
        elif self.file.mimetype in ['.csv', '.xlsx', '.xls']:
            # For structured data, typically no splitting needed
            # But return a basic splitter in case text fields need splitting
            return self._get_character_splitter()
        else:
            # Default fallback
            return self._get_recursive_splitter()
    
    def _get_recursive_splitter(self):
        """Most versatile splitter - tries paragraphs, then sentences, then characters"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _get_character_splitter(self):
        """Basic character-based splitting"""
        return CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n"
        )
    
    def _get_token_splitter(self):
        """Token-aware splitting (good for LLM context limits)"""
        return TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def _get_nltk_splitter(self):
        """NLTK-based sentence-aware splitting"""
        return NLTKTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def _get_spacy_splitter(self):
        """spaCy-based linguistic splitting"""
        return SpacyTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def _get_markdown_header_splitter(self):
        """Markdown header-aware splitting"""
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        return MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
    
    def _get_markdown_splitter(self):
        """General markdown splitting"""
        return MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def get_recommended_splitters(self):
        """Get ordered list of recommended splitters for this file type"""
        recommendations = {
            '.txt': ["recursive", "character", "nltk", "token"],
            '.md': ["markdown_header", "recursive", "markdown", "character"],
            '.pdf': ["recursive", "nltk", "character", "token"],
            '.docx': ["recursive", "nltk", "character", "token"],
            '.doc': ["recursive", "nltk", "character", "token"],
            '.ppt': ["recursive", "character", "token"],
            '.pptx': ["recursive", "character", "token"],
            '.csv': ["character", "recursive"],
            '.xlsx': ["character", "recursive"],
            '.xls': ["character", "recursive"]
        }
        
        return recommendations.get(self.file.mimetype, ["recursive", "character", "token"])

class VStoreHandler:
    def __init__(self, embedding_model):
        self.vector_store = Chroma(
            persist_directory=os.environ.get('VECTOR_DB_PERSISTENT_ABSOLUTE_PATH'),
            collection_name=os.environ.get('COLLECTION_NAME'),
            embedding_function=embedding_model
        )
    
    def select(self):
        return self.vector_store


# Created By Amit Mahapatra