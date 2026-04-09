from fastapi import APIRouter
from pydantic import BaseModel

from src.kb_mocker.chains.qa_chain import run_agent

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str


@router.post("/ask",response_model=QuestionResponse)
async def ask(request: QuestionRequest) -> QuestionResponse:
    answer = await run_agent(request.question)
    return QuestionResponse(answer=answer)
