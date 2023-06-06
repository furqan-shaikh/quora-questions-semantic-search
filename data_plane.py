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

    # Set up the pinecone client
    pinecone_client = create_pinecone_client()

    results = get_nearest_questions(request=QueryRequest(question=question),
                                    index=INDEX_NAME,
                                    model=transformer_model,
                                    pinecone_client=pinecone_client,
                                    namespace=NAMESPACE)
    for result in results:
        print(result.similar_question, result.score)


if __name__ == "__main__":
    main()
