from flask import Flask, render_template, request
import os
import spacy
from resume_matcher.extract_entity import extract_name_email, extract_ner_dict, extract_text_from_pdf
from flask import send_from_directory
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

nlp_ner = spacy.load("./output/model-best")
nlp = spacy.load('en_core_web_sm')

@app.route("/")
def loadHomePage():
    print('vector shape',nlp_ner.vocab.vectors.shape)
    return render_template('rankresume.html')

@app.route("/resume/<filename>")
def view_resume(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/ranking', methods=['POST'])
def rank_resume():
    if request.method == "POST":
        jd_text = request.form["job_description"]
        jd_dict = extract_ner_dict(jd_text, nlp_ner)
        jd_json = json.dumps(jd_dict)

        files = request.files.getlist("resumes")
        results = []

        for file in files:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            resume_text = extract_text_from_pdf(file_path)
            name, email = extract_name_email(nlp, resume_text)
            resume_dict = extract_ner_dict(resume_text, nlp_ner)
            resume_json = json.dumps(resume_dict)

            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([resume_json, jd_json])
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

            skills = resume_dict.get("SKILLS", [])

            # print(resume_dict.keys())
            # print(skills)

            results.append({
                "name": name,
                "email": email,
                "similarity": similarity,
                "skills": set(skills),
                "companies": resume_dict.get("COMPANIES WORKED AT", "N/A"),
                "yoe": resume_dict.get("YEARS OF EXPERIENCE", "N/A"),
                "resume_filename": file.filename

            })

        # Sort results by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return render_template("rankresume.html", results=results)

if __name__ == '__main__':
    app.run(debug=True)