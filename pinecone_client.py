from pinecone import Client
from dataclasses import dataclass
from transformer_model import TransformerModel
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
        self.pinecone = Client(api_key=credentials.api_key, region=credentials.environment)
        self.index = None

    def create_index(self, index_create_request: IndexCreateRequest):
        # if index_create_request.index_name not in pinecone.list_indexes():
        #     pinecone.create_index(
        #         name=index_create_request.index_name,
        #         dimension=index_create_request.dimension,
        #         metric=index_create_request.metric
        #     )
        self.index = self.pinecone.Index(index_create_request.index_name)

    def get_index(self, index_name: str):
        self.index = self.pinecone.Index(index_name)
        return self.index

    def get_index_stats(self, index_name: str):
        return self.get_index(index_name).describe_index_stats()

    def index_upsert(self, vectors, namespace: str):
        self.index.upsert(vectors=vectors, namespace=namespace)

    def query(self, vector, namespace:str,top_k:int=5):
        return self.index.query(top_k=top_k,values=vector, namespace=namespace, include_metadata=True)


def create_pinecone_client(index_create_request: IndexCreateRequest):
    pinecone_client = PineconeClient(PineconeCredentials(api_key=API_KEY, environment=ENVIRONMENT))
    pinecone_client.create_index(index_create_request=index_create_request)
    return pinecone_client
