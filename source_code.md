## Hierarchy
src:.
├───__pycache__
│   └         main.cpython-312.pyc
├───ingest
│   ├         __init__.py
│   ├───__pycache__
│   │   ├         __init__.cpython-312.pyc
│   │   ├         cache.cpython-312.pyc
│   │   ├         components.cpython-312.pyc
│   │   ├         config.cpython-312.pyc
│   │   ├         executor.cpython-312.pyc
│   │   ├         pipeline.cpython-312.pyc
│   │   └         selector.cpython-312.pyc
│   ├         cache.py
│   ├         executor.py
│   ├         pipeline.py
│   └         selector.py
├         main.py
└───query
    ├───__pycache__
    │   └         pipeline.cpython-312.pyc
    └         pipeline.py

## Code
**src/main.py**
```python
from fastapi import FastAPI, UploadFile

from ingest.pipeline import add_to_ingest_pipeline
from query.pipeline import get_result_from_pipeline


app = FastAPI(debug=True)

@app.post('/')
async def ingest(file: UploadFile):
    return add_to_ingest_pipeline(object = file.file, name =file.filename, owner='test')
    


@app.delete('/')
async def remove(name: str):
    pass


@app.get('/')
async def query(query: str):
    return get_result_from_pipeline(query)
```

**src/ingest/__init__.py**
```python


# Created By Amit Mahapatra
```

**src/ingest/cache.py**
```python
from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import TextIO, BinaryIO
import shutil

from dotenv import load_dotenv


load_dotenv()


class File:
    def __init__(self, object: TextIO | BinaryIO, name):
        self.object = object # this is actually a python file object, I need a better name here
        self.name: str = name
        self._filepath: Path = None
        self._mimetype: str = None
        self._rounded_filesize: int = None

    @property
    def filepath(self):
        return self._filepath
    
    @filepath.setter
    def filepath(self, filepath):
        self._filepath = filepath

    @property
    def mimetype(self):
        return self._mimetype
    
    @mimetype.setter
    def mimetype(self, mimetype):
        self._mimetype = mimetype
    
    @property
    def filesize(self):
        return self._rounded_filesize
    
    @filesize.setter
    def filesize(self, filesize):
        self._rounded_filesize = filesize


class Command(ABC):
    @abstractmethod
    def execute(self) -> bool:
        pass


class Save(Command):
    def __init__(self, file: File):
        self.file = file

    def execute(self) -> bool:
        data_directory_absolute_path = Path(os.environ.get('DATA_DIRECTORY_ABSOLUTE_PATH'))
        try:
            with open(data_directory_absolute_path / self.file.name, 'wb') as buffer:
                self.file.object.seek(0)
                shutil.copyfileobj(fsrc=self.file.object, fdst=buffer)
            self.file.filepath = data_directory_absolute_path / self.file.name
            self.file.mimetype = self.file.filepath.suffix
            self.file.filesize = self.file.filepath.stat().st_size
            return True
        except Exception as e:
            print(e)
            return False


class Remove(Command):
    def __init__(self, file: File):
        self.file = file

    def execute(self) -> bool:
        try:
            self.file.filepath.unlink()
            return True
        except Exception as e:
            print(e)
            return False


class Commander:
    def perform(self, command: Command):
        command.execute()


# Created By Amit Mahapatra
```

**src/ingest/executor.py**
```python
from typing import List, Dict, Optional, TypedDict
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
```

**src/ingest/pipeline.py**
```python
import os
from typing import List, TextIO, BinaryIO

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from ingest.cache import Save, Remove, Commander
from ingest.selector import LoaderSelector, SplitterSelector, VStoreHandler
from ingest.executor import LoaderExecutor, SplitterExecutor, EnricherExecutor, VStoreExecutor
from ingest.cache import File
from ingest.executor import Component, PayloadType


load_dotenv()


class IngestionCachePipeline:
    def __init__(self, object, name):
        self.file = File(object=object, name=name)

class IngestionConfigurationPipeline:
    pass

class IngestionComponentPipeline:
    def __init__(self, components: List[Component]):
        self.components = components
    
    def run(self, payload: PayloadType = {}):
        for component in self.components:
            payload = component.run(payload)
        return payload

class Pipeline:
    pass


def add_to_ingest_pipeline(object: TextIO | BinaryIO, name: str, owner :str):
    # Cache Pipeline
    file = File(object=object, name=name)
    save = Save(file)
    remove = Remove(file)
    commander = Commander()
    commander.perform(save)
    

    # Selector Pipeline
    embedding_model = HuggingFaceEmbeddings(
            model_name=os.environ.get('EMBEDDING_MODEL_REPO_ID')
    )    
    selected_loader = LoaderSelector(file).select()
    selected_splitter = SplitterSelector(file).select()
    vector_store = VStoreHandler(embedding_model=embedding_model).select()
    
    # Ingestion Pipeline
    ingestion_pipeline = IngestionComponentPipeline(
        components = [
            LoaderExecutor(selected_loader),
            SplitterExecutor(selected_splitter),
            EnricherExecutor(metadata={'owner': owner}),
            VStoreExecutor(vector_store)
        ]
    )
    ingestion_pipeline.run()
    
    # Uncache Pipeline
    commander.perform(remove)


# Created By Amit Mahapatra
```

**src/ingest/selector.py**
```python
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
```

**src/query/pipeline.py**
```python
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


embedding_model = HuggingFaceEmbeddings(
        model_name=os.environ.get('EMBEDDING_MODEL_REPO_ID')
)


db = Chroma(
    persist_directory=Path(os.environ.get('VECTOR_DB_PERSISTENT_ABSOLUTE_PATH')),
    embedding_function=embedding_model,
    collection_name=os.environ.get('COLLECTION_NAME')
)


retriever = db.as_retriever(
    search_type='similarity',
    k=5
)


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


prompt_creation_chain = RunnableParallel(
    {
        'context': retriever | RunnableLambda(lambda docs: '\n\n'.join( [ doc.page_content for doc in docs ] )),
        'query': RunnablePassthrough()
    }
)

chain = prompt_creation_chain | prompt_template | chat_model | output_parser

def get_result_from_pipeline(query: str):
    return chain.invoke(query)

# Created By Amit Mahapatra
```

