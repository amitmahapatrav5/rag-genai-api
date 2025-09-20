from fastapi import FastAPI, UploadFile

from ingest.pipeline import add_to_ingest_pipeline
from query.pipeline import get_result_from_pipeline


app = FastAPI()

@app.post('/')
async def ingest(file: UploadFile):
    add_to_ingest_pipeline(file.file)


@app.delete('/')
async def remove(name: str):
    pass


@app.get('/')
async def query(query: str):
    get_result_from_pipeline(query)