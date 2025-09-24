import os
import sys
import json
import subprocess
import time
import boto3
from urllib.parse import urlparse
import shutil
import base64
import requests
import re
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# --- Configuration ---
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET_NAME = os.environ.get('AWS_S3_BUCKET_NAME')
JOB_ID = os.environ.get('JOB_ID')
FFMPEG_PATH = "ffmpeg"
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
PEXELS_API_KEY = os.environ.get('PEXELS_API_KEY')
FLASK_APP_URL = f"http://{os.environ.get('FLASK_VM_IP')}:5000"

IMAGE_MODELS_TO_TRY = [
    "imagen-4.0-fast-generate-preview-06-06",
    "imagen-3.0-fast-generate-001", "imagen-3.0-generate-002", "imagegeneration@006"
]

# --- Status Reporting & Helper Functions ---
def update_status(job_id, message, error=False):
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    status_data = { "status": "failed" if error else "in_progress", "message": message, "timestamp": time.time() }
    s3_client.put_object( Bucket=AWS_S3_BUCKET_NAME, Key=f"jobs/{job_id}/status.json", Body=json.dumps(status_data), ContentType='application/json' )
    print(f"Status update: {message}")

def get_stock_footage(api_key, query, output_path, is_short_form=False):
    print(f"Searching for stock footage with query: '{query}'")
    if not api_key:
        print("Error: PEXELS_API_KEY is missing from the environment.")
        return None
    try:
        headers = {"Authorization": api_key}
        orientation = "portrait" if is_short_form else "landscape"
        params = { "query": query, "per_page": 5, "orientation": orientation }
        
        api_url = "https://api.pexels.com/videos/search"
        response = requests.get(api_url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        
        search_result = response.json()
        
        if not search_result.get("videos"):
            print(f"No stock footage found for '{query}'.")
            return None

        best_video = search_result["videos"][0]
        video_files = best_video.get("video_files", [])
        
        best_file = None
        for f in sorted(video_files, key=lambda x: x.get('width', 0), reverse=True):
            if f.get('width') and f['width'] <= 1920:
                best_file = f
                break
        if not best_file and video_files: best_file = video_files[0]

        if not best_file or not best_file.get("link"):
            print("Could not find a suitable video file link.")
            return None

        video_response = requests.get(best_file["link"], stream=True)
        video_response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in video_response.iter_content(chunk_size=8192): f.write(chunk)
        print(f"Successfully downloaded stock footage to {output_path}")
        return output_path
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("Error: Pexels API key is invalid or unauthorized. Please check your GitHub Actions secrets.")
        else:
            print(f"HTTP Error fetching stock footage: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching stock footage: {e}")
        return None

def upload_to_s3(file_path, object_name):
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    s3_client.upload_file(file_path, AWS_S3_BUCKET_NAME, object_name)
    print(f"Uploaded {os.path.basename(file_path)} to S3.")

def generate_script_from_prompt(api_key, prompt):
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(endpoint, headers=headers, json=data, timeout=90)
    response.raise_for_status()
    json_text = response.json()['candidates'][0]['content']['parts'][0]['text']
    return json.loads(json_text.strip().replace('```json', '').replace('```', ''))["script"]["scenes"]

def generate_image_with_retries(api_key, prompt, output_path, is_short_form=False):
    print(f"Generating image for prompt: '{prompt}'")
    aspect_ratio = "9:16" if is_short_form else "16:9"
    for model_name in IMAGE_MODELS_TO_TRY:
        try:
            vertexai.init(project=GCP_PROJECT_ID)
            model = ImageGenerationModel.from_pretrained(model_name)
            images = model.generate_images(prompt=prompt, number_of_images=1, aspect_ratio=aspect_ratio, negative_prompt="text, letters, words, watermark, signature, logo")
            if images:
                images[0].save(location=output_path)
                print(f"Successfully generated image with {model_name}.")
                return
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
    raise Exception("Image generation failed.")

def generate_audio(api_key, text, voice_name, output_path):
    tts_service = build('texttospeech', 'v1', developerKey=api_key)
    request_body = {'input': {'text': text}, 'voice': {'languageCode': voice_name[:5], 'name': voice_name}, 'audioConfig': {'audioEncoding': 'MP3'}}
    response = tts_service.text().synthesize(body=request_body).execute()
    with open(output_path, 'wb') as out:
        out.write(base64.b64decode(response['audioContent']))
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', output_path], stdout=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

def process_url_content(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    soup = BeautifulSoup(requests.get(url, headers=headers).content, 'html.parser')
    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']): element.decompose()
    return soup.get_text(separator=' ', strip=True)

def call_update_code_usage(code):
    try:
        requests.post(f"{FLASK_APP_URL}/update-code-usage", json={"code": code}, timeout=10).raise_for_status()
    except Exception as e:
        print(f"WARNING: Failed to update code usage for '{code}'. Error: {e}")

def generate_dynamic_prompt(content, channel_name, narrator_style, is_short_form):
    style = f"a '{narrator_style}' style" if narrator_style else "a neutral, informative style"
    scenes = "8 to 12" if is_short_form else "15 to 20"
    intro = f'Start with: "Welcome back to {channel_name}!"' if channel_name else "Start with a compelling hook."
    outro = f'End with: "Thanks for watching {channel_name}!"' if channel_name else "End with a concluding thought."
    return f"""
    ACT as a YouTube host with {style}. Create a script from the content below for a video with {scenes} scenes.
    {intro} {outro}
    INPUT: [{content}]
    TASK: Return a valid JSON object: {{"script": {{"title": "A viral-style title", "scenes": [{{"line": "Spoken line...", "image_prompt": "Detailed visual description for AI. No text.", "video_search_query": "2-3 word search query for stock footage."}}]}}}}
    """

def process_job(job_data):
    job_id = job_data.get('job_id')
    print(f"\n🚀 Starting job: {job_id}")
    update_status(job_id, "Worker started...")
    input_dir, output_dir = f"./{job_id}_input", f"./{job_id}_output"
    os.makedirs(input_dir, exist_ok=True); os.makedirs(output_dir, exist_ok=True)
    
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    try:
        api_key, pexels_api_key, url, preview_code, channel_name, narrator_style, is_short_form, visual_source, voice_settings, caption_settings, bg_music_url = (
            job_data.get(k) for k in ['api_key', 'pexels_api_key', 'url', 'preview_code', 'channel_name', 'narrator_style', 'is_short_form', 'visual_source', 'voice_settings', 'caption_settings', 'background_music_url'])
        
        update_status(job_id, "Scraping URL...")
        content = process_url_content(url)
        update_status(job_id, "Generating script...")
        prompt = generate_dynamic_prompt(content, channel_name, narrator_style, is_short_form)
        script = generate_script_from_prompt(api_key, prompt)
        if not script: raise Exception("Failed to generate script.")
        
        assets = []
        for i, scene in enumerate(script):
            update_status(job_id, f"Generating assets for scene {i+1}/{len(script)}...")
            visual_path = None
            if visual_source == 'stock_video':
                query = scene.get('video_search_query', 'abstract')
                visual_path = get_stock_footage(pexels_api_key, query, os.path.join(input_dir, f"scene_{i}.mp4"), is_short_form)
            
            if not visual_path:
                visual_path = os.path.join(input_dir, f"scene_{i}.png")
                generate_image_with_retries(api_key, scene.get('image_prompt'), visual_path, is_short_form)
            
            audio_path = os.path.join(input_dir, f"scene_{i}.mp3")
            duration = generate_audio(api_key, scene.get('line'), voice_settings.get('narrator'), audio_path)
            assets.append({'duration': duration, 'visual': visual_path, 'audio': audio_path})
        
        update_status(job_id, "Rendering video clips...")
        clips = []
        for i, asset in enumerate(assets):
            clip_path = os.path.join(output_dir, f"scene_{i}.mp4")
            is_video_asset = asset['visual'].endswith('.mp4')
            res_wh = "720x1280" if is_short_form else "1280x720"
            res_colon = "720:1280" if is_short_form else "1280:720"
            cmd = [FFMPEG_PATH, '-y']
            if is_video_asset: cmd.extend(['-i', asset['visual']])
            else: cmd.extend(['-loop', '1', '-r', '24', '-i', asset['visual']])
            cmd.extend(['-i', asset['audio']])
            filter_v = f"[0:v]scale={res_colon}:force_original_aspect_ratio=increase,crop={res_colon},setsar=1"
            if not is_video_asset:
                frames = int(asset['duration'] * 24); filter_v += f",zoompan=z='min(zoom+0.0005,1.1)':d={frames}:s={res_wh}"
            cmd.extend(['-filter_complex', filter_v, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', '-t', str(asset['duration']), clip_path])
            subprocess.run(cmd, check=True, capture_output=True)
            clips.append(clip_path)

        update_status(job_id, "Stitching video clips together...")
        concat_list = os.path.join(output_dir, "concat.txt")
        with open(concat_list, 'w') as f:
            for c in clips: f.write(f"file '{os.path.basename(c)}'\n")

        video_no_music_path = os.path.join(output_dir, "final_no_music.mp4")
        subprocess.run([FFMPEG_PATH, '-y', '-f', 'concat', '-safe', '0', '-i', concat_list, '-c', 'copy', video_no_music_path], check=True, capture_output=True)

        final_video_path = os.path.join(output_dir, "final_video.mp4")

        if bg_music_url:
            update_status(job_id, "Downloading and mixing background music...")
            local_bg_music_path = os.path.join(input_dir, os.path.basename(urlparse(bg_music_url).path))
            s3.download_file(AWS_S3_BUCKET_NAME, urlparse(bg_music_url).path.lstrip('/'), local_bg_music_path)
            
            music_volume = int(caption_settings.get('musicVolume', 15)) / 100.0
            
            mix_filter = f"[1:a]volume={music_volume}[bga];[0:a][bga]amix=inputs=2:duration=first[a]"
            
            ffmpeg_mix_cmd = [
                FFMPEG_PATH, '-y', '-i', video_no_music_path, '-i', local_bg_music_path,
                '-filter_complex', mix_filter, '-map', '0:v', '-map', '[a]',
                '-c:v', 'copy', '-c:a', 'aac', '-shortest', final_video_path
            ]
            subprocess.run(ffmpeg_mix_cmd, check=True, capture_output=True)
        else:
            os.rename(video_no_music_path, final_video_path)
            
        update_status(job_id, "Finalizing and uploading video...")
        upload_to_s3(final_video_path, f"jobs/{job_id}/output/final_video.mp4")
        if preview_code: call_update_code_usage(preview_code)
        print(f"✅ Job {job_id} complete!")

    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        if isinstance(e, subprocess.CalledProcessError):
            error_message += f"\nFFMPEG STDERR: {e.stderr.decode() if e.stderr else 'No stderr'}"
        update_status(job_id, error_message, error=True)
        sys.exit(1)
    finally:
        if os.path.exists(input_dir): shutil.rmtree(input_dir)
        if os.path.exists(output_dir): shutil.rmtree(output_dir)

if __name__ == "__main__":
    if not JOB_ID: sys.exit(1)
    s3_main = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    job_path = f"/tmp/{JOB_ID}.json"
    s3_main.download_file(AWS_S3_BUCKET_NAME, f"jobs/{JOB_ID}/job.json", job_path)
    with open(job_path) as f:
        process_job(json.load(f))

