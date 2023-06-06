from dataset_readers import get_quora_questions
from transformer_model import TransformerModel
from vector_embedings import create_vector_embeddings_batched
from pinecone_client import create_pinecone_client
from constants import INDEX_NAME, NAMESPACE


def main():
    # load the dataset
    questions = get_quora_questions()

    # # # create the model
    transformer_model = TransformerModel()

    # Set up the pinecone client
    pinecone_client = create_pinecone_client()

    # save the embeddings
    create_vector_embeddings_batched(questions=questions, model=transformer_model, pinecone_client=pinecone_client,
                                     namespace=NAMESPACE, index_name=INDEX_NAME)


if __name__ == "__main__":
    main()
