def extract_text_from_pdfs(pdf_paths):
    texts = []
    for path in pdf_paths:
        with fitz.open(path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            texts.append(text)
    return texts