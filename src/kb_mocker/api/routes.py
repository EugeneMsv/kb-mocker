from fastapi import APIRouter
from pydantic import BaseModel

from src.kb_mocker.chains.qa_chain import run_agent

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    reasoning: str
    answer: str


@router.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest) -> QuestionResponse:
    result = await run_agent(request.question)
    return QuestionResponse(reasoning=result.reasoning, answer=result.answer)
