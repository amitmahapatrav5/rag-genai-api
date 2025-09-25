from fastapi import FastAPI, UploadFile

from ingest.pipeline import add_to_ingest_pipeline
from query.pipeline import get_result_from_pipeline


app = FastAPI(debug=True)

@app.post('/ingest')
async def ingest(file: UploadFile):
    status = add_to_ingest_pipeline(object = file.file, name =file.filename, owner='test')
    return {"status" : status}
    


@app.delete('/')
async def remove(name: str):
    pass


@app.get('/query')
async def query(query: str):
    return get_result_from_pipeline(query)


# Created By Amit Mahapatra