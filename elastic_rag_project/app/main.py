from fastapi import FastAPI
from api import endpoints


app = FastAPI()
app.include_router(endpoints.router)

def main():
    return {"message": "Connection successful!"}
