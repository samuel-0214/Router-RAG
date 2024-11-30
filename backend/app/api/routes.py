from fastapi import APIRouter, HTTPException
from app.rag.models import Query, RAGResponse
from app.rag.core import RAGSystem

router = APIRouter()
rag_system = RAGSystem()

@router.post("/query", response_model=RAGResponse)
async def query_documents(query: Query):
    try:
        return await rag_system.process_query(query.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
