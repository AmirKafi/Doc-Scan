import fastapi

app = fastapi.FastAPI()

@app.post('OMR')
def omr_process():
    pass