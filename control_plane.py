from dataset_readers import get_quora_questions
from transformer_model import TransformerModel
from vector_embedings import create_vector_embeddings_file, create_vector_embeddings_batched
from pinecone_client import create_pinecone_client, IndexCreateRequest
from constants import COSINE_METRIC, INDEX_NAME, NAMESPACE
import json



def main():
    # load the dataset
    questions = get_quora_questions()

    # # # create the model
    transformer_model = TransformerModel()

    # # Setup the pinecone client
    pinecone_client = create_pinecone_client(
        index_create_request=IndexCreateRequest(
            index_name=INDEX_NAME, 
            dimension=transformer_model.get_sentence_embedding_dimension(),
            metric=COSINE_METRIC))

    # # save the embeddings to file
    # create_vector_embeddings_file("data/questions_data.json",questions, transformer_model, namespace='quora_questions')
    create_vector_embeddings_batched(questions=questions, model=transformer_model,pinecone_client=pinecone_client, namespace=NAMESPACE)

if __name__ == "__main__":
    main()
