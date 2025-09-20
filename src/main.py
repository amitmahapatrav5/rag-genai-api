from fastapi import FastAPI
from fastapi import UploadFile

app = FastAPI()

@app.get('/vectorize')
async def vectorize(file: UploadFile):
    pass
