import os
import ffmpeg
from flask import Flask, request, send_from_directory, jsonify
from openai import OpenAI
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Serve index.html directly
@app.route("/")
def home():
    return send_from_directory(".", "index.html")

# Serve static files
@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)

@app.route("/translate", methods=["POST"])
def translate():
    try:
        target_lang = request.form.get("language")
        video_file = request.files.get("video")

        if not video_file:
            return jsonify({"error": "No video uploaded"}), 400

        filename = secure_filename(video_file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        video_file.save(input_path)

        # Extract audio
        audio_path = input_path.replace(".mp4", ".wav")
        ffmpeg.input(input_path).output(audio_path).run(overwrite_output=True)

        # Transcribe audio
        with open(audio_path, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio
            )

        original_text = transcript.text

        # Translate text
        translation = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate the text."},
                {"role": "user", "content": f"Translate to {target_lang}: {original_text}"}
            ]
        )

        translated_text = translation.choices[0].message["content"]

        # Generate TTS
        audio_out = input_path.replace(".mp4", f"_{target_lang}.wav")

        tts_audio = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=translated_text
        )

        with open(audio_out, "wb") as f:
            f.write(tts_audio.read())

        # Merge video + dubbed audio (system command to avoid duplicate keyword issue)
        output_video = f"dubbed_{target_lang}_{filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_video)

        merge_cmd = f'ffmpeg -i "{input_path}" -i "{audio_out}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
        os.system(merge_cmd)

        return jsonify({
            "status": "success",
            "output_video": f"/static/output/{output_video}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/static/output/<path:filename>")
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
