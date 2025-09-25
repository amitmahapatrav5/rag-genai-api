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
    return True


# Created By Amit Mahapatra