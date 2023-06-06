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


def get_nearest_questions(request: QueryRequest, model: TransformerModel, pinecone_client: PineconeClient,
                          namespace: str, index: str):
    # get the vector embedding for the question
    xc = model.create_vector_embedding(request.question).tolist()
    # send the request to pine client
    results = pinecone_client.query(index_name=index, vector=xc, namespace=namespace, top_k=request.count)
    responses = []
    for result in results.matches:
        responses.append(QueryResponse(id=result.id, score=result.score, similar_question=result.metadata['text']))
    return responses
