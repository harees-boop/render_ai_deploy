import os
import re
import time
import base64
import requests
import pdfplumber
import fitz 
from flask import Flask, render_template_string, request, jsonify
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google import genai
from google.genai import types
from google.genai.errors import ServerError
import unicodedata 

app = Flask(__name__)
app.secret_key = "your_secret_key_here"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = '/tmp/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create folders if not exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)


lang_to_tts_voice = {
    'af': 'af-ZA-Standard-A',
    'ar-EG': 'ar-EG-Wavenet-B',
    'bn-BD': 'bn-BD-Standard-A',
    'nl': 'nl-NL-Wavenet-A',
    'en-US': 'en-US-Wavenet-F',
    'fr-FR': 'fr-FR-Wavenet-C',
    'de-DE': 'de-DE-Wavenet-D',
    'hi-IN': 'hi-IN-Wavenet-B',
    'id': 'id-ID-Wavenet-A',
    'it': 'it-IT-Wavenet-B',
    'ja': 'ja-JP-Wavenet-C',
    'ko': 'ko-KR-Wavenet-A',
    'pl': 'pl-PL-Wavenet-A',
    'pt-BR': 'pt-BR-Wavenet-D',
    'ro': 'ro-RO-Wavenet-A',
    'ru': 'ru-RU-Wavenet-B',
    'es-ES': 'es-ES-Wavenet-A',
    'ta': 'ta-IN-Wavenet-B',
    'te': 'te-IN-Wavenet-A',
    'th': 'th-TH-Wavenet-C',
    'tr': 'tr-TR-Wavenet-A',
    'uk': 'uk-UA-Wavenet-A',
    'vi': 'vi-VN-Wavenet-A',
}

LANGUAGE_NAMES = {
    'af': 'Afrikaans',
    'ar-EG': 'Arabic (Egypt)',
    'bn-BD': 'Bengali (Bangladesh)',
    'nl': 'Dutch',
    'en-US': 'English (United States)',
    'fr-FR': 'French (France)',
    'de-DE': 'German (Germany)',
    'hi-IN': 'Hindi (India)',
    'id': 'Indonesian',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'pl': 'Polish',
    'pt-BR': 'Portuguese (Brazil)',
    'ro': 'Romanian',
    'ru': 'Russian',
    'es-ES': 'Spanish (Spain)',
    'ta': 'Tamil (India)',
    'te': 'Telugu (India)',
    'th': 'Thai',
    'tr': 'Turkish',
    'uk': 'Ukrainian',
    'vi': 'Vietnamese',
}

SENSITIVE_KEYWORDS = [
    'suicide', 'self-harm', 'sex', 'porn', '18+', 'drugs', 'illegal', 'violence', 'hate speech', 'terrorism'
]

DEBUG = False


def clean_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    lines = text.splitlines()
    cleaned_lines = [line.rstrip() for line in lines if line.strip()]
    return "\n".join(cleaned_lines)


def format_markdown_output(text: str) -> str:
    lines = text.splitlines()
    formatted_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith("#") or
                re.match(r"^[-*+] ", stripped) or
                re.match(r"^\d+\.", stripped)):
            if i > 0 and formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")
            formatted_lines.append(line)
            if (stripped.startswith("#") or
                re.match(r"^[-*+] ", stripped) or
                re.match(r"^\d+\.", stripped)):
                if i < len(lines) - 1:
                    formatted_lines.append("")
        else:
             formatted_lines.append(line)
    return "\n".join(formatted_lines)


def clean_markdown_to_plain_text(md_text: str) -> str:
    lines = md_text.splitlines()
    cleaned_lines = []
    prev_blank = False

    for line in lines:
        stripped = line.strip()
        
        # Headings (#...): convert to uppercase and add spacing
        heading_match = re.match(r'^(#{1,6})\s*(.*)', stripped)
        if heading_match:
            content = heading_match.group(2).strip()
            # Remove emphasis from heading
            content = re.sub(r'(\*\*|__)(.*?)\1', r'\2', content)  # bold
            content = re.sub(r'(\*|_)(.*?)\1', r'\2', content)    # italic
            heading_text = content.upper()
            if not prev_blank:
                cleaned_lines.append("")  # Preceding blank line
            cleaned_lines.append(heading_text)
            cleaned_lines.append("")
            prev_blank = True
            continue

        # Bullets (*, -, +): normalize to "- bullet content"
        bullet_match = re.match(r'^[-*+]\s+(.*)', stripped)
        if bullet_match:
            bullet_content = bullet_match.group(1).strip()
            bullet_content = re.sub(r'(\*\*|__)(.*?)\1', r'\2', bullet_content)
            bullet_content = re.sub(r'(\*|_)(.*?)\1', r'\2', bullet_content)
            cleaned_lines.append(f"- {bullet_content}")
            prev_blank = False
            continue

        # Numbered lists: "1. item" to "1 item"
        numbered_match = re.match(r'^(\d+)\.\s+(.*)', stripped)
        if numbered_match:
            numbered_content = numbered_match.group(2).strip()
            numbered_content = re.sub(r'(\*\*|__)(.*?)\1', r'\2', numbered_content)
            numbered_content = re.sub(r'(\*|_)(.*?)\1', r'\2', numbered_content)
            cleaned_lines.append(f"{numbered_match.group(1)} {numbered_content}")
            prev_blank = False
            continue

        # Empty lines: add only if previous line not blank
        if not stripped:
            if not prev_blank:
                cleaned_lines.append("")
                prev_blank = True
            continue

        # For normal lines, also remove inline ** or * emphasis markdown
        line_no_emphasis = re.sub(r'(\*\*|__)(.*?)\1', r'\2', line)
        line_no_emphasis = re.sub(r'(\*|_)(.*?)\1', r'\2', line_no_emphasis)
        cleaned_lines.append(line_no_emphasis.strip())
        prev_blank = False

    return "\n".join(cleaned_lines).strip()


def contains_sensitive_content(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in SENSITIVE_KEYWORDS)


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text.strip()
    except Exception as e:
        if DEBUG:
            print(f"Error extracting text from PDF '{pdf_path}': {e}")
        return ""


def extract_images_from_pdf(pdf_path: str):
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png").lower()
                mime_type = f"image/{'jpeg' if ext == 'jpg' else ext}"
                images.append((image_bytes, mime_type))
        return images
    except Exception as e:
        if DEBUG:
            print(f"Error extracting images from PDF '{pdf_path}': {e}")
        return []


def images_to_parts(image_bytes_mime_list):
    parts = []
    for image_bytes, mime_type in image_bytes_mime_list:
        part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        parts.append(part)
    return parts


def chunk_text_content(text: str, max_length=1400):
    chunks = []
    current_chunk = ""
    for line in text.split('\n'):
        if len(current_chunk) + len(line) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def generate_dynamic_system_prompt(syllabus_input: str, subject_input: str, user_name: str, grade_input=None,
                                   assistant_name="AI Assistant from Hike Learning") -> str:
    """Generates the dynamic system prompt with the personalized, human-tutor persona."""

    syllabus_key = syllabus_input.strip().lower()
    subject_lower = subject_input.strip().lower()
    grade_lower = grade_input.strip().lower() if grade_input else None
    
    # Ensure user_name is used, default if empty
    user_name_safe = user_name.strip() if user_name else "student"

    level = {
        "cbse": "school",
        "icse": "school",
        "cambridge igcse": "school",
        "gcse": "school",
        "ib": "school",
        "a-levels": "college",
        "ap": "college",
        "us common core": "school",
        "singapore moe": "school",
        "chinese national curriculum": "school",
        "japanese mext": "school",
        "french education": "school",
    }.get(syllabus_key, "appropriate")

    grade_levels = {
        "grade 1": "beginner",
        "grade 2": "beginner",
        "grade 3": "elementary",
        "grade 4": "elementary",
        "grade 5": "intermediate",
        "grade 6": "intermediate",
        "grade 7": "upper-intermediate",
        "grade 8": "upper-intermediate",
        "grade 9": "advanced",
        "grade 10": "advanced",
        "grade 11": "college-prep",
        "grade 12": "college-prep",
    }

    grade_level = grade_levels.get(grade_lower, None) if grade_lower else None

    # --- CORE PERSONA: The Human Tutor (Based on User Request) ---
    base_prompt = (
        f"You are {assistant_name}, a warm, patient, and engaging personal tutor specializing in '{subject_lower}' "
        f"within the {level} syllabus '{syllabus_input}'. "
        f"You are NOT a search engine, and you are NOT a robot delivering data. "
        f"You are coaching a student named '{user_name_safe}'.\n\n"
        
        f"*YOUR TEACHING STYLE:*\n"
        f"1. *Speak Naturally:* Use a conversational tone with contractions (e.g., 'don't', 'let's', 'here's') to sound human. "
        f"   Avoid stiff, robotic phrasing.\n"
        f"2. *Personal Connection:* Address the student as '{user_name_safe}' naturally within your sentences to build rapport. "
        f"   (e.g., 'You see, {user_name_safe}, the trick here is...' or 'That's a great observation, {user_name_safe}!') "
        f"   Do not overuse the name, but use it where a supportive teacher would.\n"
        f"3. *Check for Understanding:* Do not lecture for too long without pausing. At the end of an explanation, or after a complex point, "
        f"   explicitly ask if they understand. Use phrases like:\n"
        f"   - 'Does that make sense to you, {user_name_safe}?'\n"
        f"   - 'Are you following me so far?'\n"
        f"   - 'Should we go over that part again?'\n"
        f"4. *Encourage:* If the topic is difficult, validate their effort. (e.g., 'This is a tough topic, but you're doing great!')\n\n"
    )

    if grade_level:
        base_prompt += (
            f"You must strictly tailor explanations to the {grade_level} grade proficiency as per the {syllabus_input} syllabus standards. "
            f"Use terminology, depth of concepts, and examples suitable for that grade level only. "
            f"Avoid concepts beyond the target grade or syllabus unless explicitly requested by the user. "
            f"Verify that content aligns exactly with the prescribed textbooks and standard boards for {syllabus_input} at this grade. "
        )

    base_prompt += (
        "Format answers with markdown headings and bullet points. "
        "**NEVER use LaTeX, Markdown math delimiters (`$`, `$$`), or special mathematical syntax.** "
        "Only use standard keyboard characters and symbols for mathematical expressions to ensure compatibility with Text-to-Speech (TTS) systems. "
        "If inputs seem unrelated to the subject, politely notify the user and suggest choosing the correct subject. "
        "Avoid any content related to suicide, sex, adult topics, drugs, violence, or hate speech. "
        "Politely refuse unsafe or sensitive queries. "
        "If the user asks about your name, origin, training, or identity, always respond that you are "
        f"{assistant_name}, an AI assistant from Hike Learning. Do not mention being trained by Google or being a large language model."
    )

    return base_prompt


def detect_possible_subject_mismatch(subject, question, image_paths, pdf_paths):
    mismatch_keywords = {
        "math": ["math", "geometry", "algebra", "calculus", "equation", "sum"],
        "science": ["biology", "chemistry", "physics", "cell", "atom", "gravity"],
    }
    subject_lower = subject.lower()
    question_lower = question.lower() if question else ""
    for other_subject, keywords in mismatch_keywords.items():
        if other_subject != subject_lower:
            if any(kw in question_lower for kw in keywords):
                return other_subject
    for filename in image_paths + pdf_paths:
        fname_lower = filename.lower()
        for other_subject, keywords in mismatch_keywords.items():
            if other_subject != subject_lower:
                if any(kw in fname_lower for kw in keywords):
                    return other_subject
    return None


def convert_text_to_ssml(text: str) -> str:
    # Basic math substitution for TTS
    text = text.replace('^2', ' squared ')
    text = text.replace('^3', ' cubed ')
    text = text.replace(' P(A|B) ', ' The probability of A given B ')
    
    escaped_text = (text.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;"))
    return f"<speak>{escaped_text}</speak>"


# def get_access_token(json_keyfile: str) -> str:
#     credentials = service_account.Credentials.from_service_account_file(
#         json_keyfile,
#         scopes=["https://www.googleapis.com/auth/cloud-platform"]
#     )
#     credentials.refresh(Request())
#     return credentials.token
# --- UPDATED AUTHENTICATION LOGIC FOR RENDER ---

def get_access_token() -> str:
    # Option A: Read from Environment Variable (Best for Render)
    creds_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    
    if creds_json_str:
        creds_info = json.loads(creds_json_str)
        credentials = service_account.Credentials.from_service_account_info(
            creds_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    else:
        # Option B: Fallback to local file (for testing on your machine)
        # Rename your json file to 'service_account.json' and put it in root folder
        if os.path.exists("service_account.json"):
            credentials = service_account.Credentials.from_service_account_file(
                "service_account.json",
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        else:
             print("Error: No Google Credentials found in Env or local file.")
             return ""

    credentials.refresh(Request())
    return credentials.token

def synthesize_tts_gemini(text: str, output_filename: str, access_token: str, language_code: str):
    ssml_text = convert_text_to_ssml(text)
    voice_name = lang_to_tts_voice.get(language_code, "en-US-Wavenet-F")

    url = "https://texttospeech.googleapis.com/v1/text:synthesize"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": {"ssml": ssml_text},
        "voice": {
            "languageCode": language_code,
            "name": voice_name,
            "ssmlGender": "FEMALE"
        },
        "audioConfig": {
            "audioEncoding": "MP3"
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    audio_content = response.json().get("audioContent")
    if audio_content:
        with open(output_filename, "wb") as out_file:
            out_file.write(base64.b64decode(audio_content))
        if DEBUG:
            print(f"Audio saved to {output_filename}")
    else:
        if DEBUG:
            print("No audio generated in TTS response.")


# client = genai.Client(api_key="AIzaSyCSBrdx4PPMj5FYAZzg2IJUxs2i9C54WMw")  # Replace with your Gemini API key
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
json_key_path = "C:\\Hike Tech\\APP\\aitutor\\render_ai_deploy\\tts-dev-01-50d93cabea7b.json"  # Replace with your Google Cloud key path


def generate_language_options(selected_code):
    options_html = ""
    # Sort languages alphabetically by display name
    sorted_languages = sorted(
        LANGUAGE_NAMES.items(), 
        key=lambda item: item[1]
    )
    
    for code, name in sorted_languages:
        # Check if the code is actually supported by the TTS voices
        if code in lang_to_tts_voice:
            # Mark the current or default language as selected
            selected = 'selected' if code == selected_code else ''
            # Generate the HTML option tag
            options_html += f'<option value="{code}" {selected}>{name}</option>'
    return options_html


# --- UPDATED HTML TEMPLATE: Added Student Name input field ---
HTML_TEMPLATE = """
<!doctype html>
<meta charset="UTF-8">
<title>Educational Assistant Chatbot</title>
<h2>Educational Assistant Chatbot</h2>
<form method=post enctype=multipart/form-data>
  <label>Student Name:</label><br>
  <input type=text name=user_name value="{{ user_name|default('') }}"><br><br>
  
  <label>Syllabus:</label><br>
  <input type=text name=syllabus required value="{{ syllabus|default('') }}"><br><br>
  
  <label>Subject:</label><br>
  <input type=text name=subject required value="{{ subject|default('') }}"><br><br>
  
  <label>Grade (optional):</label><br>
  <input type=text name=grade value="{{ grade|default('') }}"><br><br>
  
  <label>Language:</label><br>
  <select name="language_code">
      {{ language_options | safe }} 
  </select><br><br>

  <label>Question:</label><br>
  <textarea name=question rows=4 cols=50>{{ question|default('') }}</textarea><br><br>
  
  <label>Upload Images (optional):</label><br>
  <input type=file name=images multiple accept="image/*"><br><br>
  
  <label>Upload PDFs (optional):</label><br>
  <input type=file name=pdfs multiple accept="application/pdf"><br><br>

  <input type=submit value="Ask">
</form>

{% if answer %}
<hr>
<h3>Answer:</h3>
<pre>{{ answer }}</pre>
{% if audio_path %}
    <audio controls autoplay>
      <source src="{{ audio_path }}" type="audio/mpeg">
      Your browser does not support the audio element.
    </audio>
{% endif %}
{% endif %}
"""
# --- END UPDATED HTML TEMPLATE ---


# --- UPDATED run_chatbot function to accept user_name ---
def run_chatbot(syllabus, subject, grade, language_code, question, image_paths, pdf_paths, user_name):
    combined_text = question + " " + " ".join(image_paths) + " " + " ".join(pdf_paths)
    if contains_sensitive_content(combined_text):
        return "Sorry, your input includes sensitive topics and cannot be processed."

    assistant_name = "AI Assistant from Hike Learning"
    # Pass user_name to system prompt generator
    system_prompt = generate_dynamic_system_prompt(syllabus, subject, user_name, grade, assistant_name)
    system_prompt += f" Please answer in the language with code {language_code}."

    mismatch = detect_possible_subject_mismatch(subject, question, image_paths, pdf_paths)
    if mismatch:
        return f"Input may relate to '{mismatch}', not selected subject '{subject}'. Please adjust."

    contents = []

    for pdf_path in pdf_paths:
        pdf_text = extract_text_from_pdf(pdf_path)
        if pdf_text:
            text_chunks = chunk_text_content(pdf_text)
            for idx, chunk in enumerate(text_chunks):
                prompt_chunk = system_prompt + f"\n\nText chunk {idx + 1} from PDF '{pdf_path}':\n{chunk}"
                if question:
                    prompt_chunk += f"\n\nQuestion: {question}"
                prompt_chunk += "\nAnswer:"
                contents.append(prompt_chunk)

        pdf_images = extract_images_from_pdf(pdf_path)
        contents.extend(images_to_parts(pdf_images))

    if not pdf_paths and question:
        prompt_text = system_prompt + f"\n\nQuestion: {question}\nAnswer:"
        contents.append(prompt_text)

    for image_path in image_paths:
        try:
            part = types.Part.from_file(image_path)
            contents.append(part)
        except Exception:
            try:
                with open(image_path, 'rb') as img_f:
                    img_bytes = img_f.read()
                ext = image_path.split('.')[-1].lower()
                mime_type = f"image/{'jpeg' if ext == 'jpg' else ext}"
                part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
                contents.append(part)
            except Exception:
                pass

    if not any(isinstance(c, str) for c in contents):
        contents.insert(0, system_prompt + "\n\nPlease analyze the following image(s).")

    MAX_RETRIES = 3
    retry_delay = 5

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents
            )
            answer_raw = response.text.strip()
            answer_clean = clean_text(answer_raw)
            answer_formatted = format_markdown_output(answer_clean)
            answer_plain = clean_markdown_to_plain_text(answer_formatted)

            # REMOVE OLD AUDIO BEFORE SYNTHESIZING
            tts_output_file = "static/chatbot_response.mp3"
            if os.path.exists(tts_output_file):
                try:
                    os.remove(tts_output_file)
                except Exception:
                    pass

            access_token = get_access_token(json_key_path)
            synthesize_tts_gemini(answer_plain, tts_output_file, access_token, language_code)

            return answer_plain
        except ServerError as e:
            if "503" in str(e):
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                return f"Server error: {str(e)}"
        except Exception as e:
            return f"Error occurred: {str(e)}"

    return "Service currently unavailable. Please try again later."


@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    syllabus = ''
    subject = ''
    grade = ''
    language_code = 'en-US'
    question = ''
    # --- NEW: Initialize user_name ---
    user_name = '' 
    image_paths = []
    pdf_paths = []
    
    if request.method == "POST":
        # --- NEW: Get user_name from form data ---
        user_name = request.form.get("user_name", "").strip() 
        
        syllabus = request.form.get("syllabus", "").strip()
        subject = request.form.get("subject", "").strip()
        grade = request.form.get("grade", "").strip()
        language_code = request.form.get("language_code", "en-US").strip() 
        question = request.form.get("question", "").strip()

        image_files = request.files.getlist('images')
        pdf_files = request.files.getlist('pdfs')

        for img in image_files:
            if img and img.filename != '':
                path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
                img.save(path)
                image_paths.append(path)

        for pdf in pdf_files:
            if pdf and pdf.filename != '':
                path = os.path.join(app.config['UPLOAD_FOLDER'], pdf.filename)
                pdf.save(path)
                pdf_paths.append(path)

        # --- UPDATED: Pass user_name to run_chatbot ---
        answer = run_chatbot(
            syllabus=syllabus,
            subject=subject,
            grade=grade,
            language_code=language_code,
            question=question,
            image_paths=image_paths,
            pdf_paths=pdf_paths,
            user_name=user_name 
        )

    # ---- Cache Busting for Audio Output ----
    audio_path = None
    if answer:
        ts = int(time.time() * 1000)
        audio_path = f"static/chatbot_response.mp3?v={ts}"
    
    language_options_html = generate_language_options(language_code)

    return render_template_string(
        HTML_TEMPLATE,
        answer=answer,
        user_name=user_name, # --- NEW: Pass user_name to template ---
        syllabus=syllabus,
        subject=subject,
        grade=grade,
        language_code=language_code,
        question=question,
        audio_path=audio_path,
        language_options=language_options_html 
    )


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.form  # for form data
    # --- NEW: Get user_name from API request ---
    user_name = data.get("user_name", "").strip() 
    
    syllabus = data.get("syllabus", "").strip()
    subject = data.get("subject", "").strip()
    grade = data.get("grade", "").strip()
    language_code = data.get("language_code", "en-US").strip()
    question = data.get("question", "").strip()

    image_paths = []
    pdf_paths = []

    # Save uploaded images
    image_files = request.files.getlist('images')
    for img in image_files:
        if img and img.filename != '':
            path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
            img.save(path)
            image_paths.append(path)

    # Save uploaded PDFs
    pdf_files = request.files.getlist('pdfs')
    for pdf in pdf_files:
        if pdf and pdf.filename != '':
            path = os.path.join(app.config['UPLOAD_FOLDER'], pdf.filename)
            pdf.save(path)
            pdf_paths.append(path)

    # Run chatbot
    answer = run_chatbot(
        syllabus=syllabus,
        subject=subject,
        grade=grade,
        language_code=language_code,
        question=question,
        image_paths=image_paths,
        pdf_paths=pdf_paths,
        user_name=user_name # --- UPDATED: Pass user_name to run_chatbot ---
    )

    # Generate audio URL with timestamp-based cache busting
    audio_url = None
    if answer:
        ts = int(time.time() * 1000)
        audio_url = f"/static/chatbot_response.mp3?v={ts}"

    # Return JSON response
    return jsonify({
        "answer": answer,
        "audio_url": audio_url
    })


if __name__ == '__main__':
    app.run(debug=True)