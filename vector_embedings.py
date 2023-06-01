# Creates vector embeddings in a JSON file
from dataclasses import dataclass, field
from typing import Any, List
from transformer_model import TransformerModel
from pinecone_client import PineconeClient
import json
from tqdm.auto import tqdm
from constants import INDEX_NAME


@dataclass
class VectorCustom:
    id: str
    metadata: Any
    values: Any


@dataclass
class Vectors:
    vectors: List[VectorCustom] = field(default_factory=list)


def create_vector_embeddings_file(file_name: str, questions: List[str], model: TransformerModel, namespace: str):
    index = 1
    vectors = Vectors()
    print("Creating vector embeddings for: " + str(len(questions)) + " questions")
    for question in questions:
        vector_embedding = model.create_vector_embedding(question)
        index = index + 1
        vector = VectorCustom(id=str(index), metadata={'query': question}, values=vector_embedding.tolist())
        vectors.vectors.append(vector)
        print("Created Vector embedding for question: " + str(index))

    # save to a json file
    print("Saving to json file")
    json_string = json.dumps({'vectors': [ob.__dict__ for ob in vectors.vectors], 'namespace': namespace})
    write_file(file_name, json_string)


def write_file(file_name, data):
    with open(file_name, "w") as f:
        f.write(data)


def create_vector_embeddings_batched(questions: List[str], model: TransformerModel, pinecone_client: PineconeClient,
                                     namespace: str):
    batch_size = 128
    total_batches = int(len(questions) / batch_size)
    print("Total number of batches: " + str(total_batches))
    for i in tqdm(range(0, len(questions), batch_size)):
        # find end of batch
        i_end = min(i + batch_size, len(questions))
        print("Starting Batch: " + str(i) + " Start: " + str(i) + " End: " + str(i_end))
        records = []
        for x in range(i, i_end):
            xc = model.create_vector_embedding(questions[x])
            records.append((str(x), xc, {'text': questions[x]}))
        # # upsert to Pinecone
        pinecone_client.index_upsert(vectors=records, namespace=namespace)
        print("Completed Batch: " + str(i) + " Start: " + str(i) + " End: " + str(i_end))

    # check number of records in the index
    print(pinecone_client.get_index_stats(INDEX_NAME))

    # Vector; Tuple[str, List[float]]; Tuple[str, List[float], dict]; Dict[str, Any]`
