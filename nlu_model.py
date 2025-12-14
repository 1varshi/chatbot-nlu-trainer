import spacy

# Load pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

def analyze_text(text):
    doc = nlp(text)

    # spaCy has entities but NO intent classifier
    # We set intent as 'general' for now
    intent = "general"

    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })

    return {
        "intent": intent,
        "entities": entities
    }

