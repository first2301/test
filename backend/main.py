from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.generate_router import router as generate_router
from api.chat_router import router as chat_router

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate_router, tags=['generate']) #  prefix="/generate",
app.include_router(chat_router, tags=['chat']) #  prefix="/chat",

app.get('/')
def main():
    return "connection success!"