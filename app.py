import spacy
from PyPDF2 import PdfReader
from docx import Document
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from flask import Flask, render_template, request
import PyPDF2
import fitz 
import os


#upload_folder = os.path.join(app.root_path, 'uploads')

# Ensure the upload folder exists, or create it if it doesn't
#os.makedirs(upload_folder, exist_ok=True)


# Load the English NLP model from spaCy
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# Sample labeled data (You should have a larger and more diverse labeled dataset)
data = [
    ("Strong Python and machine learning skills.", "Selected"),
    ("Project manager with excellent communication.", "Selected"),
    ("No relevant skills or experience.", "Rejected"),
]

# Split the data into features (resume text) and labels (Selected/Rejected)
X = [text for text, label in data]
y = [label for text, label in data]

# Create a TF-IDF vectorizer to convert text into numerical features
tfidf_vectorizer = TfidfVectorizer()

# Create a Multinomial Naive Bayes classifier
classifier = MultinomialNB()

# Create a pipeline that combines the vectorizer and classifier
model = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('classifier', classifier)
])

# Train the model on the labeled data
model.fit(X, y)

def classify_and_score_resume(resume_text, primary_skills, secondary_skills, primary_weight, secondary_weight, threshold):
    # Process the resume text using spaCy
    doc = nlp(resume_text)

    # Calculate the score for primary skills
    primary_score = sum(skill.lower() in doc.text.lower() for skill in primary_skills) / len(primary_skills)

    # Calculate the score for secondary skills
    secondary_score = sum(skill.lower() in doc.text.lower() for skill in secondary_skills) / len(secondary_skills)

    # Calculate the weighted total score
    total_score = (primary_score * primary_weight) + (secondary_score * secondary_weight)

    # Determine the result based on the total_score and threshold
    result = "Selected" if total_score >= threshold else "Rejected"

    return result, total_score


def extract_required_skills(job_description):
    # Process the job description using spaCy
    doc = nlp(job_description)

    # Extract nouns and noun phrases as required skills
    required_skills = [token.text for token in doc if token.pos_ in ['NOUN', 'NOUN_CHUNK']]
    return required_skills

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    total_score = 0.0  # Initialize total_score with a default value
    primary_skills = []  # Initialize primary_skills with an empty list
    secondary_skills = []  # Initialize secondary_skills with an empty list
    primary_weight = None  # Initialize primary_weight as None
    secondary_weight = None  # Initialize secondary_weight as None
    missing_skills = []  # Initialize missing_skills as an empty list

    if request.method == 'POST':
        file = request.files['resume']
        job_description = request.form['job_description']
        threshold = float(request.form['threshold'])

        # Extract primary skills and secondary skills from the form
        primary_skills = [skill.strip() for skill in request.form['primary_skills'].split(',')]
        secondary_skills = [skill.strip() for skill in request.form['secondary_skills'].split(',')]
        primary_weight = float(request.form['primary_weight'])
        secondary_weight = float(request.form['secondary_weight'])

        try:
            # Extract required skills from the job description
            required_skills = extract_required_skills(job_description)

            # Process the uploaded file and perform screening
            resume_text = extract_text_from_document(file)
            result, total_score = classify_and_score_resume(
                resume_text, primary_skills, secondary_skills, primary_weight, secondary_weight, threshold
            )

            # Calculate missing skills based on required skills and skills found in the resume
            missing_skills = [skill for skill in required_skills if skill.lower() not in resume_text.lower()]
        except Exception as e:
            return str(e)

    return render_template(
        'index.html',
        result=result,
        total_score=total_score,
        primary_skills=primary_skills,
        secondary_skills=secondary_skills,
        primary_weight=primary_weight,
        secondary_weight=secondary_weight,
        missing_skills=missing_skills
    )



def extract_text_from_document(file_obj):
    if file_obj.filename.endswith('.pdf'):
        # Extract text from PDF
        pdf_text = ''
        pdf_reader = PdfReader(file_obj)
        for page_num in range(len(pdf_reader.pages)):
            pdf_text += pdf_reader.pages[page_num].extract_text()
        return pdf_text
    elif file_obj.filename.endswith('.docx'):
        # Extract text from Word document
        doc = Document(file_obj)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format")
    

def extract_text_from_pdf(pdf_file):
    # Initialize an empty string to store extracted text
    text = ""

    # Open the PDF file using PyMuPDF (fitz)
    pdf_document = fitz.open(pdf_file)

    # Iterate through each page of the PDF
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)

        # Extract text from the page and append it to the text string
        text += page.get_text()

    return text

@app.route('/extracted_skills', methods=['GET', 'POST'])
def extracted_skills():
    extracted_skills = []

    if request.method == 'POST':
        if 'resume' in request.files:
            resume = request.files['resume']
            if resume.filename != '':
                try:
                    # Save the uploaded file to the dynamically generated upload folder
                    resume.save(os.path.join(upload_folder, resume.filename))

                    # Read the uploaded resume file
                    with open(os.path.join(upload_folder, resume.filename), 'r', encoding='utf-8') as file:
                        resume_text = file.read()

                    # Extract skills from resume_text using your skill extraction logic
                    extracted_skills = extract_skills(resume_text)
                except Exception as e:
                    return str(e)

    return render_template('extracted_skills.html', extracted_skills=extracted_skills)


@app.route('/display_skills', methods=['GET', 'POST'])
def display_skills():
    extracted_skills = []

    if request.method == 'POST':
        if 'resume' in request.files:
            resume = request.files['resume']
            if resume.filename != '':
                # Read the resume text and extract skills using your AI model
                resume_text = resume.read().decode('utf-8')
                extracted_skills = extracted_skills(resume_text)  # Replace with your skill extraction logic

    return render_template('skills.html', extracted_skills=extracted_skills)

if __name__ == '__main__':
    app.run(debug=True)
