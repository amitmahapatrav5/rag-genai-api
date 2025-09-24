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
│   │   └         pipeline.cpython-312.pyc
│   ├         cache.py
│   ├         components.py
│   ├         pipeline.py
│   └         selector.py
├         main.py
└───query
    ├───__pycache__
    │   └         pipeline.cpython-312.pyc
    ├         components.py
    ├         config.py
    ├         pipeline.py
    └         util.py

## Code
**src/main.py**
```python
from fastapi import FastAPI, UploadFile

from ingest.pipeline import add_to_ingest_pipeline
from query.pipeline import get_result_from_pipeline


app = FastAPI(debug=True)

@app.post('/')
async def ingest(file: UploadFile):
    add_to_ingest_pipeline(object = file.file, name =file.filename)
    return {'success': True} 



@app.delete('/')
async def remove(name: str):
    pass


@app.get('/')
async def query(query: str):
    get_result_from_pipeline(query)
```

**src/ingest/__init__.py**
```python


# Created By Amit Mahapatra
```

**src/ingest/cache.py**
```python
from abc import ABC, abstractmethod
import mimetypes
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
        self.encoding: str = None
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
            self.file.mimetype, self.file.encoding = mimetypes.guess_type(self.file.filepath)
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

**src/ingest/components.py**
```python
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
```

**src/ingest/pipeline.py**
```python
from typing import List, TextIO, BinaryIO

from dotenv import load_dotenv

from ingest.cache import Save, Remove, Commander
from ingest.selector import LoaderSelector, SplitterSelector, VStoreHandler
from ingest.components import Loader, Splitter, Enricher, VStore
from ingest.cache import File
from ingest.components import Component, PayloadType


load_dotenv()


class IngestionCachePipeline:
    pass

class IngestionConfigurationPipeline:
    pass

class IngestionComponentPipeline:
    def __init__(self, components: List[Component]):
        self.components = components
    
    def run(self, payload: PayloadType):
        for component in self.components:
            payload = component.run(payload)
        return payload

class Pipeline:
    pass


def add_to_ingest_pipeline(object: TextIO | BinaryIO, name: str):
    
    # Cache Pipeline
    file = File(object=object, name=name)
    save = Save(file)
    remove = Remove(file)
    commander = Commander()
    commander.perform(save)
    

    # Selector Pipeline
    
    
    # Ingestion Pipeline
    pipeline = Pipeline()
    pipeline.run()
    
    # Uncache Pipeline
    commander.perform(remove)

# Created By Amit Mahapatra
```

**src/ingest/selector.py**
```python
class LoaderSelector:
    pass


class SplitterSelector:
    pass


class VStoreHandler:
    pass


# Created By Amit Mahapatra
```

**src/query/components.py**
```python

```

**src/query/config.py**
```python

```

**src/query/pipeline.py**
```python
import os
from abc import ABC, abstractmethod
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
    collection_name=os.environ.get('VECTOR_DB_NAME')
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

**src/query/util.py**
```python

```

