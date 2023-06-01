from transformer_model import TransformerModel
from pinecone_client import create_pinecone_client, IndexCreateRequest
from constants import COSINE_METRIC, INDEX_NAME, NAMESPACE
from query_engine import get_nearest_questions, QueryRequest
import json
import sys



def main():
    question = sys.argv[1]
    # create the model
    transformer_model = TransformerModel()

    # # Setup the pinecone client
    pinecone_client = create_pinecone_client(
        index_create_request=IndexCreateRequest(
            index_name=INDEX_NAME, 
            dimension=transformer_model.get_sentence_embedding_dimension(),
            metric=COSINE_METRIC))

    results = get_nearest_questions(request = QueryRequest(question=question),
                                    model=transformer_model ,
                                    pinecone_client=pinecone_client,
                                    namespace=NAMESPACE)
    for result in results:
        print(result.similar_question, result.score)

if __name__ == "__main__":
    main()
