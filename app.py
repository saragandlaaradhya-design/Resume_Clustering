from flask import Flask, render_template, request, send_from_directory
import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists('uploads'):
    os.makedirs('uploads')


# 📄 Extract text
def extract_text(path):
    text = ""
    try:
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except:
        pass
    return text


# 🧠 Extract skills
def extract_skills(text):
    skills = [
        "python","java","c++","machine learning","ai",
        "data science","html","css","javascript",
        "react","node","sql","flask","django"
    ]
    text = text.lower()
    return [s for s in skills if s in text]


# ⭐ Score
def calculate_score(skills):
    return round((len(skills)/12)*100,2)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/cluster', methods=['GET','POST'])
def cluster():
    clusters = {}
    resume_skills = {}
    resume_scores = {}
    resume_match = {}
    ranked_resumes = []
    labels = []
    sizes = []

    if request.method == 'POST':
        files = request.files.getlist('resumes')
        job_desc = request.form.get('job_desc', '').lower()

        texts = []
        names = []

        for file in files:
            if file.filename == "":
                continue

            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)

            text = extract_text(path)
            skills = extract_skills(text)

            if skills:
                texts.append(" ".join(skills))
                names.append(file.filename)

                resume_skills[file.filename] = skills
                resume_scores[file.filename] = calculate_score(skills)

            if len(texts) < 2:
                return render_template(
                'cluster.html',
                message="⚠ Please upload at least 2 resumes",
                clusters={},
                resume_skills={},
                resume_scores={},
                resume_match={},
                ranked_resumes=[],
                labels=[],
                sizes=[]
            )

        

        else:
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(texts)

            k = min(3, len(texts))
            model = KMeans(n_clusters=k, random_state=42)
            results = model.fit_predict(X)

            cluster_names = ["AI/Data Science","Web Development","Software"]

            for i, label in enumerate(results):
                if label not in clusters:
                    clusters[label] = {
                        "name": cluster_names[label] if label < 3 else f"Cluster {label}",
                        "files":[]
                    }
                clusters[label]["files"].append(names[i])

        # 🎯 Job matching
        for name, skills in resume_skills.items():
            if job_desc:
                match = sum(1 for s in skills if s in job_desc)
                score = (match/len(skills))*100 if skills else 0
            else:
                score = 0
            resume_match[name] = round(score,2)


        ranked_resumes = sorted(resume_scores.items(), key=lambda x: x[1], reverse=True)

        
        labels = []
        sizes = []

        if clusters:
            for v in clusters.values():
             labels.append(v["name"])
             sizes.append(len(v["files"]))

    return render_template(
        'cluster.html',
        clusters=clusters,
        resume_skills=resume_skills,
        resume_scores=resume_scores,
        resume_match=resume_match,
        ranked_resumes=ranked_resumes,
        labels=labels,
        sizes=sizes
    )


# 📂 Resume preview
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)