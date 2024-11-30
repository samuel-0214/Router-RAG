import json
import openai
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from app.rag.models import Document, RAGResponse
from app.rag.utils import load_config

class RAGSystem:
    def __init__(self):
        config = load_config()
        self.client = openai.OpenAI(api_key=config["openai_api_key"])
        self.model_name = config["model_name"]
        self.embedding_model = config["embedding_model"]
        
        # Load data
        self.data = self._load_documents()
        self.documents = [str(item) for item in self.data]
        self.embeddings = self._create_embeddings(self.documents)

    def _load_documents(self) -> List[Dict]:
        with open("app/data/documents.json", "r") as f:
            return json.load(f)

    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float"
            )
            embeddings.append(response.data[0].embedding)
        return np.array(embeddings)

    def _get_embedding(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
            encoding_format="float"
        )
        return np.array(response.data[0].embedding)

    async def process_query(self, query: str) -> RAGResponse:
        # Get relevant documents
        relevant_docs = self._get_relevant_context(query)
        
        # Generate response
        answer = await self._generate_response(query, relevant_docs)
        
        return RAGResponse(
            answer=answer,
            relevant_documents=relevant_docs
        )

    def _get_relevant_context(self, query: str, k: int = 3) -> List[Document]:
        query_embedding = self._get_embedding(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        return [
            Document(
                content=self.documents[i],
                similarity_score=float(similarities[i])
            )
            for i in top_k_indices
        ]

    async def _generate_response(self, query: str, relevant_docs: List[Document]) -> str:
        context = "\n\n".join([doc.content for doc in relevant_docs])
        
        prompt = f"""Based on the following context, please answer the question.
        If the answer cannot be found in the context, say so.

        Context:
        {context}

        Question: {query}

        Answer:"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer questions based solely on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content