from sentence_transformers import SentenceTransformer
from typing import Union, List
import torch

class TransformerModel:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def create_vector_embedding(self,query: Union[str,List[str]]):
        return self.model.encode(query)

    def get_sentence_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()