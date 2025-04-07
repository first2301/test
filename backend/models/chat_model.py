from pydantic import BaseModel

class ChatRequest(BaseModel):
    model: str
    question: str

class ChatResponse(BaseModel):
    answer: str