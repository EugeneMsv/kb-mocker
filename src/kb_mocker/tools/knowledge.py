import os

from langchain_core.tools import tool

from src.kb_mocker.config import settings


@tool
def list_knowledge_files() -> list[str]:
    """
      List all markdown (.md) files available in the local knowledge base.
      Call this first when the user asks about a topic — it tells you what
      knowledge files exist so you can decide which one to load.
    """
    kb_path = settings.knowledge_base_path
    if not os.path.isdir(kb_path):
        return []

    md_files = []
    for file in os.listdir(kb_path):
        if file.endswith(".md"):
            md_files.append(file)

    return md_files


@tool
def load_knowledge(filename: str) -> str:
    """
      Load the full content of a markdown knowledge file by its exact filename
      (including .md extension). Use list_knowledge_files first to discover
      available filenames. Returns the file content as plain text.
    """
    file_path = os.path.join(settings.knowledge_base_path, filename)

    if not os.path.isfile(file_path):
        return f"File '{filename}' not found in knowledge base."

    with open(file_path) as file:
        return file.read()
