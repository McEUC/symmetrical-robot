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
from PIL import Image

# --- CONFIGURATION ---
app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- READ KEYS FROM THE ENVIRONMENT ---
# The os module is needed to read environment variables
import os 

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME")
AWS_S3_REGION = os.environ.get("AWS_S3_REGION", "us-east-2") # Default to us-east-1 if not set

# --- HELPER FUNCTIONS ---
def upload_to_s3(file_path, object_name):
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_S3_REGION)
    try:
        # We removed the ExtraArgs to be compatible with modern S3 settings
        s3_client.upload_file(file_path, AWS_S3_BUCKET_NAME, object_name)
        url = f"https://{AWS_S3_BUCKET_NAME}.s3.amazonaws.com/{object_name}"
        print(f"Successfully uploaded {object_name} to S3.")
        return url
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return None

def generate_script_and_prompts(api_key, wiki_content):
    print("Generating script with Gemini...")
    prompt = f"""
    Based on the following text from a Backrooms wiki page, create a short, atmospheric video script with 3 to 5 scenes.
    The script should have a narrator and one commentator. For each scene, provide a 'line' for the 'speaker' and a detailed, spooky, 'image_prompt'.
    The speakers can ONLY be 'narrator' or 'commenter1'.
    Respond ONLY with a valid JSON object in the format: {{"scenes": [{{"speaker": "narrator", "line": "Dialogue here.", "image_prompt": "Image prompt here."}}]}}
    Wiki Content: --- {wiki_content} ---
    """
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(endpoint, headers=headers, json=data)
    response.raise_for_status()
    json_text = response.json()['candidates'][0]['content']['parts'][0]['text']
    json_text = json_text.strip().replace('```json', '').replace('```', '')
    return json.loads(json_text).get("scenes", [])

def generate_image(api_key, prompt, output_path):
    print(f"Generating placeholder image for prompt: '{prompt}'")
    img = Image.new('RGB', (1280, 720), color='darkslategray')
    img.save(output_path)

def generate_audio(api_key, text, voice_name, output_path):
    print(f"Generating audio for: '{text}' with voice {voice_name}")
    tts_service = build('texttospeech', 'v1', developerKey=api_key)
    
    text_chunks = re.split(r'(?<=[.?!])\s+', text.strip())
    audio_clips = []
    clip_dir = os.path.dirname(output_path)
    concat_file_path = os.path.join(clip_dir, "concat_list.txt")

    for i, chunk in enumerate(text_chunks):
        if not chunk: continue
        
        request_body = { 'input': {'text': chunk}, 'voice': {'languageCode': voice_name[:5], 'name': voice_name}, 'audioConfig': {'audioEncoding': 'MP3'} }
        response = tts_service.text().synthesize(body=request_body).execute()
        
        clip_path = os.path.join(clip_dir, f"temp_audio_{i}.mp3")
        with open(clip_path, 'wb') as out:
            out.write(base64.b64decode(response['audioContent']))
        audio_clips.append(clip_path)

    if len(audio_clips) > 1:
        with open(concat_file_path, 'w') as f:
            for clip in audio_clips:
                f.write(f"file '{os.path.basename(clip)}'\n")
        ffmpeg_cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_file_path, '-c', 'copy', output_path]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    elif len(audio_clips) == 1:
        os.rename(audio_clips[0], output_path)

    for clip in audio_clips:
        if os.path.exists(clip): os.remove(clip)
    if os.path.exists(concat_file_path): os.remove(concat_file_path)

    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', output_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout)

# --- FLASK ROUTES ---
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
        content_div = soup.find('div', class_='mw-parser-output')
        wiki_text = content_div.get_text(separator=' ', strip=True)[:4000]
        script_data = generate_script_and_prompts(google_api_key, wiki_text)

        scene_assets = []
        for i, scene in enumerate(script_data):
            # --- THIS IS THE NEW, MORE ROBUST LOGIC ---
            speaker_key = scene.get('speaker')
            voice_name = voice_settings.get(speaker_key) # Use .get() to avoid errors

            # If the voice_name is missing for any reason, default to the narrator's voice
            if not voice_name:
                print(f"Warning: Invalid or missing speaker '{speaker_key}'. Defaulting to narrator voice.")
                voice_name = voice_settings.get('narrator')
            # --- END OF CORRECTION ---

            image_path = os.path.join(job_folder, f"scene_{i}.png")
            audio_path = os.path.join(job_folder, f"scene_{i}.mp3")

            generate_image(google_api_key, scene.get('image_prompt', ''), image_path)
            duration = generate_audio(google_api_key, scene.get('line', ''), voice_name, audio_path)
            scene_assets.append({'local_image': image_path, 'local_audio': audio_path, 'duration': duration, **scene})

        scene_urls = []
        for i, asset in enumerate(scene_assets):
            image_url = upload_to_s3(asset['local_image'], f"jobs/{job_id}/input/image_{i}.png")
            audio_url = upload_to_s3(asset['local_audio'], f"jobs/{job_id}/input/audio_{i}.mp3")
            if not image_url or not audio_url:
                raise Exception("Failed to upload assets to S3.")
            scene_urls.append({'image_url': image_url, 'audio_url': audio_url, **asset})

        bg_music_url = None
        if 'backgroundMusic' in request.files and request.files['backgroundMusic'].filename != '':
            music_file = request.files['backgroundMusic']
            bg_music_path = os.path.join(job_folder, "bg_music.mp3")
            music_file.save(bg_music_path)
            bg_music_url = upload_to_s3(bg_music_path, f"jobs/{job_id}/input/bg_music.mp3")

        job_data = { "job_id": job_id, "scenes": scene_urls, "caption_settings": json.loads(form_data.get('captionSettings')), "background_music_url": bg_music_url }
        job_file_path = os.path.join(job_folder, 'job.json')
        with open(job_file_path, 'w') as f:
            json.dump(job_data, f)
        upload_to_s3(job_file_path, f"jobs/{job_id}/job.json")

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