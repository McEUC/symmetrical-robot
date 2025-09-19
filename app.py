import os
import uuid
import json
import base64
import subprocess
import requests
import boto3
import re
from flask import Flask, request, jsonify, render_template
from bs4 import BeautifulSoup
from googleapiclient.discovery import build

# UPDATED: New imports for the modern Vertex AI SDK
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# --- Configuration (Unchanged) ---
app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME")
GITHUB_REPO_NAME = os.environ.get("GITHUB_REPO_NAME")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME")
AWS_S3_REGION = os.environ.get("AWS_S3_REGION", "us-east-1")


# --- Helper Functions ---
def upload_to_s3(file_path, object_name):
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_S3_REGION)
    try:
        s3_client.upload_file(file_path, AWS_S3_BUCKET_NAME, object_name)
        url = f"https://{AWS_S3_BUCKET_NAME}.s3.amazonaws.com/{object_name}"
        print(f"Successfully uploaded {object_name} to S3.")
        return url
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return None

def generate_script_and_prompts(api_key, wiki_content, article_title):
    # This function is unchanged
    print("Generating script with 'Liminal Echoes' prompt...")
    prompt = f"""
ACT as the host of 'Liminal Echoes,' a YouTube channel that explores the unsettling depths of the Backrooms. Your task is to transform a dry wiki article into a chilling, narrative-driven video script.
**Channel Style:**- **Host Persona:** You are a haunted storyteller. Your tone is hushed, conspiratorial, and filled with a sense of dread. You speak as if you are sharing forbidden knowledge that has taken a toll on you. You are not a simple narrator; you are an interpreter of the abyss.- **Content Style:** The video MUST NOT read the wiki word-for-word. Instead, you will synthesize, paraphrase, and dramatize the information. Weave the facts from the wiki into a creepy, first-person or second-person narrative. The script must be dominated by your original, unsettling commentary which should connect different pieces of information on the page to build a cohesive sense of horror.
**INPUT:**- **Wiki Article Title:** [{article_title}]- **Wiki Article Content:** [{wiki_content}]
**TASK: Structure the script EXACTLY as follows:**
1.  **Branded Intro (Speaker: narrator):** Start with the exact line: "Listen closely. Can you hear it? That hum in the static between worlds? Welcome back to Liminal Echoes, where we give voice to the silence of places that shouldn't exist."
2.  **Hook (Speaker: narrator):** Create a deeply unsettling hook (2-3 sentences) that introduces '{article_title}' by asking a disturbing question or painting a chilling mental image for the viewer.
3.  **Content Weaving Loop:** Your primary task is to process the wiki article section by section (e.g., Description, Entities, Entrances, Exits). For each section, create a two-part sequence:    -   **Information Synthesis (Speaker: narrator):** **DO NOT READ THE WIKI VERBATIM.** You must rephrase and narrate the key information from the wiki section in your own creepy style. Describe it as if you are seeing it, or as if the viewer is the one trapped there. For example, instead of 'The walls are yellow,' say 'An endless, sickening yellow stains the walls, the color of old bruises and decay...'    -   **Unsettling Analysis (Speaker: commenter1):** Immediately after synthesizing a piece of information, provide extensive, creepy, and speculative analysis. This is the heart of the video. Question the 'why'. Speculate on the malevolent intelligence behind the architecture. Connect the description of the level to the entities found within it. Discuss the psychological toll it would take. Ask disturbing rhetorical questions that linger with the viewer.
4.  **Branded Outro (Speaker: narrator):** End the video with the exact lines: "The information stops here, but the feeling doesn't. Be wary of the quiet places and the patterns you start to see in the static. The echoes are always listening. Until next time, try not to get lost."
**IMAGE PROMPT RULES (Apply to ALL prompts):**-   **Consistent Style:** Every prompt MUST include ONE of these two style phrases: 'A grainy, unsettling, found-footage style photograph' OR 'A hyper-realistic, liminal space digital painting'. The image_prompt field must always contain a string describing the scene and cannot be null.-   **Atmosphere:** Prompts should focus on creating feelings of dread, isolation, and cosmic horror. Describe empty spaces, strange architecture, distorted figures in the distance, and analog-style visual artifacts.-   **No Text:** Prompts MUST NOT contain any words, text, or letters.
Respond ONLY with a valid JSON object in the format: {{"scenes": [{{"speaker": "narrator", "line": "Dialogue.", "image_prompt": "Image prompt."}}]}}
"""
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(endpoint, headers=headers, json=data)
    response.raise_for_status()
    json_text = response.json()['candidates'][0]['content']['parts'][0]['text']
    json_text = json_text.strip().replace('```json', '').replace('```', '')
    return json.loads(json_text).get("scenes", [])

# --- UPDATED: Modern, high-level image generation function ---
def generate_image(prompt, output_path):
    print(f"Generating real image for prompt: '{prompt}'")
    
    # Initialize Vertex AI. It will use the project and location of your VM.
    # This only needs to be done once per application start, but is safe here.
    vertexai.init()
    
    model = ImageGenerationModel.from_pretrained("imagegeneration@006")
    
    images = model.generate_images(
        prompt=prompt,
        number_of_images=1,
        aspect_ratio="16:9",
        negative_prompt="text, letters, words, watermark, signature, logo"
    )
    
    # Save the generated image to the specified output path
    images[0].save(location=output_path, include_generation_parameters=False)
    print(f"Successfully saved image to {output_path}")

def generate_audio(api_key, text, voice_name, output_path):
    # This function is unchanged
    print(f"Generating audio for: '{text}' with voice {voice_name}")
    tts_service = build('texttospeech', 'v1', developerKey=api_key)
    text_chunks = re.split(r'(?<=[.?!])\s+', text.strip())
    audio_clips, concat_file_path = [], None
    clip_dir = os.path.dirname(output_path)
    for i, chunk in enumerate(text_chunks):
        if not chunk: continue
        request_body = {'input': {'text': chunk}, 'voice': {'languageCode': voice_name[:5], 'name': voice_name}, 'audioConfig': {'audioEncoding': 'MP3'}}
        response = tts_service.text().synthesize(body=request_body).execute()
        clip_path = os.path.join(clip_dir, f"temp_audio_{i}.mp3")
        with open(clip_path, 'wb') as out:
            out.write(base64.b64decode(response['audioContent']))
        audio_clips.append(clip_path)

    if len(audio_clips) > 1:
        concat_file_path = os.path.join(clip_dir, "concat_list.txt")
        with open(concat_file_path, 'w') as f:
            for clip in audio_clips: f.write(f"file '{os.path.basename(clip)}'\n")
        ffmpeg_cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_file_path, '-c', 'copy', output_path]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    elif len(audio_clips) == 1:
        os.rename(audio_clips[0], output_path)

    for clip in audio_clips:
        if os.path.exists(clip): os.remove(clip)
    if concat_file_path and os.path.exists(concat_file_path): os.remove(concat_file_path)

    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', output_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

# --- FLASK ROUTES (Unchanged from last version) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-video', methods=['POST'])
def handle_video_generation():
    try:
        form_data = request.form
        google_api_key = form_data.get('apiKey')
        voice_settings = json.loads(form_data.get('voiceSettings'))
        job_id = str(uuid.uuid4())
        job_folder = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
        os.makedirs(job_folder, exist_ok=True)

        page = requests.get(form_data.get('url'))
        soup = BeautifulSoup(page.content, 'html.parser')
        article_title = soup.title.string.replace(" Wiki | Fandom", "").strip()
        content_div = soup.find('div', class_='mw-parser-output')
        wiki_text = content_div.get_text(separator=' ', strip=True)[:6000]
        script_data = generate_script_and_prompts(google_api_key, wiki_text, article_title)

        scene_assets = []
        for i, scene in enumerate(script_data):
            speaker_key = scene.get('speaker', 'narrator').lower()
            voice_name = voice_settings.get(speaker_key)
            if not voice_name:
                print(f"Warning: Invalid speaker '{speaker_key}'. Defaulting to narrator.")
                voice_name = voice_settings.get('narrator')

            image_path = os.path.join(job_folder, f"scene_{i}.png")
            audio_path = os.path.join(job_folder, f"scene_{i}.mp3")
            generate_image(scene.get('image_prompt', 'An empty, dimly lit room.'), image_path)
            duration = generate_audio(google_api_key, scene.get('line', ''), voice_name, audio_path)
            scene_assets.append({'local_image': image_path, 'local_audio': audio_path, 'duration': duration, **scene})

        scene_urls = []
        for asset in scene_assets:
            image_url = upload_to_s3(asset['local_image'], f"jobs/{job_id}/input/{os.path.basename(asset['local_image'])}")
            audio_url = upload_to_s3(asset['local_audio'], f"jobs/{job_id}/input/{os.path.basename(asset['local_audio'])}")
            if not image_url or not audio_url: raise Exception("Failed to upload assets to S3.")
            scene_urls.append({'image_url': image_url, 'audio_url': audio_url, **asset})

        bg_music_url = None
        if 'backgroundMusic' in request.files and request.files['backgroundMusic'].filename != '':
            music_file = request.files['backgroundMusic']
            bg_music_path = os.path.join(job_folder, "bg_music.mp3")
            music_file.save(bg_music_path)
            bg_music_url = upload_to_s3(bg_music_path, f"jobs/{job_id}/input/bg_music.mp3")

        job_data = {"job_id": job_id, "scenes": scene_urls, "caption_settings": json.loads(form_data.get('captionSettings')), "background_music_url": bg_music_url}
        job_file_path = os.path.join(job_folder, 'job.json')
        with open(job_file_path, 'w') as f: json.dump(job_data, f)
        upload_to_s3(job_file_path, f"jobs/{job_id}/job.json")

        print(f"Triggering GitHub Action for job: {job_id}")
        headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {GITHUB_TOKEN}", "X-GitHub-Api-Version": "2022-11-28"}
        data = {"event_type": "video-job", "client_payload": { "job_id": job_id }}
        dispatch_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO_NAME}/dispatches"
        api_response = requests.post(dispatch_url, json=data, headers=headers)
        api_response.raise_for_status()
        print("GitHub Action triggered successfully.")
        
        return jsonify({"message": "Job submitted successfully!", "jobId": job_id})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    try:
        video_object_key = f"jobs/{job_id}/output/final_video.mp4"
        s3_client.head_object(Bucket=AWS_S3_BUCKET_NAME, Key=video_object_key)
        video_url = f"https://{AWS_S3_BUCKET_NAME}.s3.amazonaws.com/{video_object_key}"
        return jsonify({"status": "done", "downloadUrl": video_url})
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return jsonify({"status": "pending"})
        else:
            return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)