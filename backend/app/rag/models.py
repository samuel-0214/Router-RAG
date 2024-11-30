from pydantic import BaseModel
from typing import List

class Query(BaseModel):
    question: str

class Document(BaseModel):
    content: str
    similarity_score: float

class RAGResponse(BaseModel):
    answer: str
    relevant_documents: List[Document]