import re
import fitz
def extract_text_from_pdf(path):
    with fitz.open(path) as doc:
        return " ".join(page.get_text() for page in doc)

def extract_name_email(nlp, text):
    doc = nlp(text)
    name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "N/A")
    email = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return name, email.group() if email else "N/A"

def extract_ner_dict(text, model):
    doc = model(text)
    d = {}
    for ent in doc.ents:
        d.setdefault(ent.label_, []).append(ent.text)
    return d
