import pinecone
from dataclasses import dataclass
from constants import API_KEY, ENVIRONMENT


@dataclass
class PineconeCredentials:
    api_key: str
    environment: str


@dataclass
class IndexCreateRequest:
    index_name: str
    dimension: int
    metric: str


# index
#   namespace
#    vectors
class PineconeClient:
    def __init__(self, credentials: PineconeCredentials):
        pinecone.init(api_key=credentials.api_key, environment=credentials.environment)

    def index_upsert(self, vectors, namespace: str, index_name: str):
        index = pinecone.Index(index_name)
        index.upsert(vectors=vectors, namespace=namespace)

    def query(self, index_name: str, vector, namespace: str, top_k: int = 5):
        index = pinecone.Index(index_name)
        return index.query(top_k=top_k, vector=vector, namespace=namespace, include_metadata=True)


def create_pinecone_client():
    pinecone_client = PineconeClient(PineconeCredentials(api_key=API_KEY, environment=ENVIRONMENT))
    return pinecone_client
