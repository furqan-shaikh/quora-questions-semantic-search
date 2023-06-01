from datasets import load_dataset

def get_dataset(dataset: str):
    dataset = load_dataset(dataset, split='train')
    return dataset

def get_quora_dataset():
    return get_dataset('quora')

def get_quora_dataset_head():
    dataset = get_quora_dataset()
    return dataset[:5]

def get_quora_questions():
    questions_list = []
    questions = get_quora_dataset()['questions']
    for item in questions:
        text_items = item['text']
        for text in text_items:
            questions_list.append(text)
    return list(set(questions_list))

questions = get_quora_questions()

