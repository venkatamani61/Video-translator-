import os
import ffmpeg
from flask import Flask, render_template, request, send_from_directory
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "static/output"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

os.makedirs("uploads", exist_ok=True)
os.makedirs("static/output", exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        target_lang = request.form.get("language")
        video_file = request.files["video"]

        if not video_file:
            return "No video uploaded!"

        # Save uploaded video
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], video_file.filename)
        video_file.save(input_path)

        # Extract audio from video
        audio_path = input_path.replace(".mp4", ".wav")
        ffmpeg.input(input_path).output(audio_path).run(overwrite_output=True)

        # Transcribe audio (Hindi → text)
        with open(audio_path, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio
            )

        text = transcript.text

        # Translate text (Hindi → Telugu/Hindi/English/etc)
        translation = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You translate subtitles."},
                {"role": "user", "content": f"Translate this to {target_lang}: {text}"}
            ]
        )

        translated_text = translation.choices[0].message["content"]

        # Convert translated text → Speech
        audio_out = input_path.replace(".mp4", f"_{target_lang}.wav")

        tts_audio = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=translated_text
        )

        with open(audio_out, "wb") as f:
            f.write(tts_audio.read())

        # Final output path
        output_path = os.path.join(
            app.config["OUTPUT_FOLDER"],
            f"dubbed_{target_lang}_{video_file.filename}"
        )

        # FIXED MERGE (no repeated map args)
        merge_cmd = f'ffmpeg -i "{input_path}" -i "{audio_out}" -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac "{output_path}"'
        os.system(merge_cmd)

        return render_template("index.html", output_video=output_path)

    return render_template("index.html")


@app.route("/static/output/<path:filename>")
def download_file(filename):
    return send_from_directory("static/output", filename)


if __name__ == "__main__":
    app.run(debug=True)
        
