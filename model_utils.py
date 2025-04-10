from transformers import pipeline

# Load a pre-trained transformer model for NER (can replace with your custom one)
def load_ner_model():
    return pipeline("ner", grouped_entities=True, model="dslim/bert-base-NER")

def extract_entities(text, ner_model):
    entities = ner_model(text)
    results = {}
    for ent in entities:
        label = ent['entity_group']
        word = ent['word']
        results.setdefault(label, []).append(word)
    return results
