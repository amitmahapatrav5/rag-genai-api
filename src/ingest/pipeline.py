from typing import List, TextIO, BinaryIO

from dotenv import load_dotenv

from ingest.components import Component, PayloadType


load_dotenv()


class Pipeline:
    def __init__(self, components: List[Component]):
        self.components = components
    
    def run(self, payload: PayloadType):
        for component in self.components:
            payload = component.run(payload)
        return payload


def add_to_ingest_pipeline(file: TextIO | BinaryIO):
    try:
        pipeline = Pipeline()
        pipeline.run()
        return True
    except Exception as e:
        return False


# Created By Amit Mahapatra