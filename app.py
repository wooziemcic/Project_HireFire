from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from transformers import pipeline
import speech_recognition as sr
from pydub import AudioSegment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import base64
import io
import string

# Initialize Flask app and database
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///interviews.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize the FLAN-T5 model and speech recognizer
generator = pipeline('text2text-generation', model='google/flan-t5-large')
recognizer = sr.Recognizer()
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
nltk.download('stopwords')

# Database model for storing interview data
class Interview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_description = db.Column(db.Text, nullable=False)
    questions = db.Column(db.Text, nullable=False)
    transcription = db.Column(db.Text, nullable=True)
    score = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(10), nullable=True)

# Create the database tables
with app.app_context():
    db.create_all()

def preprocess_text(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def score_transcription(transcription, job_description):
    transcription_clean = preprocess_text(transcription)
    job_description_clean = preprocess_text(job_description)
    vectorizer = TfidfVectorizer().fit_transform([job_description_clean, transcription_clean])
    vectors = vectorizer.toarray()
    similarity_score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return similarity_score

def generate_questions(description):
    questions = []
    for question_type in ['technical', 'non-technical']:
        prompt = f"Based on the following job description:\n\n{description}\n\nGenerate one {question_type} interview question."
        result = generator(
            prompt, max_new_tokens=50, num_return_sequences=1,
            temperature=0.7, repetition_penalty=1.2, num_beams=5
        )[0]['generated_text']
        questions.append(result.strip())
    return questions

def transcribe_audio(audio_data):
    try:
        audio_bytes = base64.b64decode(audio_data)
        audio_buffer = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(audio_buffer, format="webm")
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        with sr.AudioFile(wav_buffer) as source:
            audio_content = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_content)
            return transcription
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError as e:
        return f"Error with transcription service: {e}"
    except Exception as e:
        print("Error during transcription:", e)
        return "An error occurred during transcription."

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/post_job', methods=['GET', 'POST'])
def post_job():
    if request.method == 'POST':
        job_description = request.form['job_description']
        questions = generate_questions(job_description)
        questions_str = "\n".join(questions)  # Store questions as a string in the database
        new_interview = Interview(job_description=job_description, questions=questions_str)
        db.session.add(new_interview)
        db.session.commit()
        return render_template('questions.html', questions=questions, job_description=job_description, interview_id=new_interview.id)
    return render_template('post_job.html')

@app.route('/record_answer/<int:interview_id>', methods=['POST'])
def record_answer(interview_id):
    interview = Interview.query.get_or_404(interview_id)
    audio_data = request.form['audio_data']
    transcription = transcribe_audio(audio_data) if audio_data else "No answer provided."
    score = score_transcription(transcription, interview.job_description)
    status = "Hired" if score >= 0.5 else "Not Hired"

    # Update the interview record in the database
    interview.transcription = transcription
    interview.score = score
    interview.status = status
    db.session.commit()

    return render_template('result.html', questions=interview.questions.split("\n"), transcription=transcription, score=score, status=status)

if __name__ == "__main__":
    app.run(debug=True)
