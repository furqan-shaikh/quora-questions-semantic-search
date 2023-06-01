from dataclasses import dataclass, field
from typing import Any, List
from transformer_model import TransformerModel
from pinecone_client import PineconeClient

@dataclass
class QueryRequest:
    question: str
    count: int = 5

@dataclass
class QueryResponse:
    similar_question: str
    score: float
    id: str

def get_nearest_questions(request: QueryRequest,model: TransformerModel, pinecone_client: PineconeClient,namespace: str):
    # get the vector embedding for the question
    xc = model.create_vector_embedding(request.question)
    # send the request to pine client
    results = pinecone_client.query(vector=xc, namespace=namespace, top_k=request.count)
    response = []
    for result in results:
        response.append(QueryResponse(similar_question=result.metadata['text'],
        score=result.score,
        id=result.id))
    return response
                                     

