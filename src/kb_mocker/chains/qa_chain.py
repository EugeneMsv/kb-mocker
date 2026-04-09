import logging
from email.policy import default
from pyexpat.errors import messages

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import tools_condition
from pydantic import BaseModel, Field

from src.kb_mocker.config import settings
from src.kb_mocker.tools.knowledge import list_knowledge_files, load_knowledge

tools =[list_knowledge_files, load_knowledge]

TOOL_MAP = {tool.name : tool for tool in tools}

logger = logging.getLogger(__name__)

system_prompt = ("You are a helpful assistant with access to a local knowledge base of markdown files. Use the "
                 "available tools to look up relevant information before answering.")

class FinalAnswer(BaseModel):
    answer: str = Field(description="The complete, concise answer to the user's question")

def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )


async def run_agent(question: str) -> str:
    llm_with_tools = _build_llm().bind_tools(tools)

    messages = [
        {"role": "system", "content": system_prompt},
        HumanMessage(content=question),
    ]

    logger.info("Starting agent loop, question=%r", question)
    response = None
    for _ in range(settings.max_iterations):
        logger.info("Iteration %d/%d", _ + 1, settings.max_iterations)
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        finish_reason = response.response_metadata.get("finish_reason")
        logger.info("finish_reason=%s", finish_reason)

        if finish_reason == "stop":
            break

        if finish_reason == "tool_calls":

            for tool_call in response.tool_calls:
                tool_fn = TOOL_MAP[tool_call["name"]]
                tool_result = tool_fn.invoke(tool_call["args"])
                logger.info("Tool %s args=%s → %s", tool_call["name"], tool_call["args"], tool_result)
                messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                )
    else:
        logger.info("Max iterations reached (%d)", settings.max_iterations)

    raw_content = response.content

    if isinstance(raw_content, list):
        raw_content = " ".join(
            block["text"] for block in raw_content if block.get("type") == "text"
        )

    return await format_response(raw_content)


async def format_response(raw_content:str ) -> str:
    formatting_prompt = ChatPromptTemplate.from_messages([
        ("system", "Return the following answer in the requested structured format. Do not change its meaning."),
        ("human", "{raw_answer}"),
    ])

    formatting_chain = formatting_prompt | _build_llm().with_structured_output(FinalAnswer)
    formatted: FinalAnswer = await formatting_chain.ainvoke({"raw_answer": raw_content})
    logger.info("Formatted response: %s", formatted.answer)
    return formatted.answer
