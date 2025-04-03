from fastapi import APIRouter
from models.request_model import QueryRequest
from services.llm_service import run_llama_inference#, stream_llama_response
from fastapi.responses import StreamingResponse

router = APIRouter()

@router.post("/generate")# generate
def generate_text(request: QueryRequest):
    """프롬프트 받아서 응답 생성"""
    output = run_llama_inference(request.prompt)
    return {"response": output}

# @router.post("/generate-stream")
# def generate_text_stream(request: QueryRequest):
#     """스트리밍 응답"""
#     return StreamingResponse(stream_llama_response(request.prompt), media_type="text/plain")