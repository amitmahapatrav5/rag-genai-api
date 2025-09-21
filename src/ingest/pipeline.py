from typing import List, TextIO, BinaryIO

from dotenv import load_dotenv

from ingest.cache import Save, Remove, Commander
from ingest.config import LoaderSelector, SplitterSelector, VStoreHandler
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
    file = File(object=object, name=name)

    save = Save(file)
    remove = Remove(file)

    commander = Commander()
    commander.perform(save)
    # commander.perform(remove)

    # try:
    #     pipeline = Pipeline()
    #     pipeline.run()
    #     return True
    # except Exception as e:
    #     return False


# Created By Amit Mahapatra