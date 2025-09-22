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
# The public IP of your VM is needed for the worker to call back to the Flask app
FLASK_APP_URL = f"http://{os.environ.get('FLASK_VM_IP')}:5000"

# List of image generation models to try in sequence
IMAGE_MODELS_TO_TRY = [
    "imagegeneration@006",
    "imagen-3.0-fast-generate-001",
    "imagen-3.0-generate-002",
    "imagen-4.0-fast-generate-preview-06-06"
]

# --- Status Reporting Function ---
def update_status(job_id, message, error=False):
    """Creates or updates a status.json file in S3 for the job."""
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    status_data = {
        "status": "failed" if error else "in_progress",
        "message": message,
        "timestamp": time.time()
    }
    status_key = f"jobs/{job_id}/status.json"
    s3_client.put_object(
        Bucket=AWS_S3_BUCKET_NAME,
        Key=status_key,
        Body=json.dumps(status_data),
        ContentType='application/json'
    )
    print(f"Status update: {message}")

# --- HELPER FUNCTIONS ---
def upload_to_s3(file_path, object_name):
    """Uploads a file to an S3 bucket and returns its public URL."""
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    s3_client.upload_file(file_path, AWS_S3_BUCKET_NAME, object_name)
    url = f"https://{AWS_S3_BUCKET_NAME}.s3.amazonaws.com/{object_name}"
    print(f"Uploaded {os.path.basename(file_path)} to S3: {url}")
    return url

def generate_script_from_prompt(api_key, prompt):
    """Generic function to call the Gemini API with a given prompt and handle retries."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            data = {"contents": [{"parts": [{"text": prompt}]}]}
            response = requests.post(endpoint, headers=headers, json=data, timeout=90)
            response.raise_for_status()
            
            response_json = response.json()
            if response_json.get('candidates'):
                json_text = response_json['candidates'][0]['content']['parts'][0]['text']
                json_text = json_text.strip().replace('```json', '').replace('```', '')
                parsed_json = json.loads(json_text)
                
                if "script" in parsed_json and "scenes" in parsed_json["script"]:
                    return parsed_json["script"]["scenes"]
                return parsed_json.get("scenes", [])

            print(f"Warning: Gemini API returned no candidates. Attempt {attempt + 1}/{max_retries}.")
            if attempt < max_retries - 1: time.sleep(60)
        except Exception as e:
            print(f"Gemini API call failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1: time.sleep(60)
            else: raise e
                
    raise Exception("Failed to get valid script from Gemini API after multiple attempts.")

def generate_image_with_retries(api_key, prompt, output_path, is_short_form=False):
    """
    Generates an image using a list of Vertex AI models, cycling through them on failure.
    """
    print(f"Generating image for prompt: '{prompt}'")
    aspect_ratio = "9:16" if is_short_form else "16:9"
    
    # Loop through the models twice to give each a second chance
    models_to_attempt = IMAGE_MODELS_TO_TRY * 2

    for model_name in models_to_attempt:
        try:
            print(f"Attempting image generation with model: {model_name}")
            vertexai.init(project=GCP_PROJECT_ID)
            model = ImageGenerationModel.from_pretrained(model_name)
            images = model.generate_images(
                prompt=prompt,
                number_of_images=1,
                aspect_ratio=aspect_ratio,
                negative_prompt="text, letters, words, watermark, signature, logo"
            )
            if images:
                images[0].save(location=output_path, include_generation_parameters=False)
                print(f"Successfully generated image with {model_name}.")
                return # Success, exit the function
            else:
                print(f"Model {model_name} returned no images.")
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            # Immediately try the next model
    
    # If all attempts fail
    raise Exception(f"Image generation failed for prompt '{prompt}' after trying all models.")

def generate_audio(api_key, text, voice_name, output_path):
    """Generates audio for a given line of text, handling long text by splitting it."""
    print(f"Generating audio for: '{text}' with voice {voice_name}")
    tts_service = build('texttospeech', 'v1', developerKey=api_key)
    text_chunks = re.split(r'(?<=[.?!])\s+', text.strip())
    audio_clips = []
    concat_file_path = None
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

# --- DYNAMIC PROMPT GENERATION ---
def generate_dynamic_prompt(url_type, content, channel_name, narrator_style, is_short_form):
    
    style_instruction = f"- **Host Persona:** You are a narrator with a '{narrator_style}' style." if narrator_style else "- **Host Persona:** You are a neutral, informative narrator."
    
    branded_intro = f'Start with the line: "Welcome back to {channel_name}!"' if channel_name else "Start with a compelling hook (1-2 sentences)."
    branded_outro = f'End with: "Thanks for watching {channel_name}! Subscribe for more!"' if channel_name else "End with a concluding thought."
    
    scene_count = "8 to 12 scenes" if is_short_form else "15 to 20 scenes"
    
    image_style_map = {
        'reddit': "'A vibrant and detailed digital illustration in a classic Japanese anime style.'",
        'backrooms': "'A grainy, unsettling, found-footage style photograph' OR 'A hyper-realistic, liminal space digital painting.'",
        'generic': "'A cinematic, hyper-realistic digital painting with dramatic lighting.'"
    }
    image_style = image_style_map.get(url_type, image_style_map['generic'])

    prompt = f"""
ACT as a YouTube host. Your task is to create an engaging video script based on the provided web content.
The script should be structured for a video with {scene_count}.

**Channel & Host Style:**
{style_instruction}
- **Content Style:** Add significant original commentary. Analyze the situation and offer a strong opinion.

**INPUT:**
- **Web Content:** [{content}]

**TASK: Structure the script EXACTLY as follows:**
1.  **Intro:** {branded_intro}
2.  **Main Story Reading (Speaker: narrator):** Break the main content into multiple segments.
3.  **Post-Story Commentary (Speaker: commenter1):** After reading, provide detailed commentary.
4.  **Outro:** {branded_outro}

**IMAGE PROMPT RULES:**
- **Style:** Every prompt MUST include one of these styles: {image_style}
- **No Text:** Prompts MUST NOT contain any words or text.

**OUTPUT FORMAT:**
Return a valid JSON object strictly following this schema:
{{
  "script": {{
    "title": "A catchy, viral-style title for the video",
    "scenes": [
      {{ "speaker": "narrator" | "commenter1", "line": "...", "image_prompt": "..." }}
    ]
  }}
}}
"""
    return prompt

# --- URL PROCESSING FUNCTION ---
def process_url_content(url):
    """Universal function to scrape content from any URL."""
    print(f"Processing URL: {url}")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    page = requests.get(url, headers=headers, timeout=15)
    page.raise_for_status()
    soup = BeautifulSoup(page.content, 'html.parser')
    
    title_tag = soup.find('h1') or soup.find('h2') or soup.find('title')
    title = title_tag.get_text(strip=True) if title_tag else 'No Title Found'
    
    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        element.decompose()
    body_text = soup.get_text(separator=' ', strip=True)
    
    full_text = f"Title: {title}. Content: {body_text}"
    print(f"Extracted content: {full_text[:300]}...")
    return full_text

def call_update_code_usage(code):
    """Calls the Flask app to increment the usage count for a code."""
    try:
        update_url = f"{FLASK_APP_URL}/update-code-usage"
        response = requests.post(update_url, json={"code": code}, timeout=10)
        response.raise_for_status()
        print(f"Successfully updated usage for code: {code}")
    except Exception as e:
        print(f"WARNING: Failed to update code usage for '{code}'. Error: {e}")

# --- Core Video Processing Function ---
def process_job(job_data):
    job_id = job_data.get('job_id')
    print(f"\n🚀 Starting job: {job_id}")
    update_status(job_id, "Worker started, preparing assets...")

    input_dir = f"./{job_id}_input"
    output_dir = f"./{job_id}_output"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    
    try:
        api_key = job_data.get('api_key')
        url = job_data.get('url')
        preview_code = job_data.get('preview_code')
        channel_name = job_data.get('channel_name')
        narrator_style = job_data.get('narrator_style')
        is_short_form = job_data.get('is_short_form', False)
        
        try:
            update_status(job_id, "Scraping the URL...")
            scraped_content = process_url_content(url)
        except Exception as e:
            update_status(job_id, f"Error: Could not read URL. {e}", error=True)
            raise

        update_status(job_id, "Generating script with AI...")
        url_type = 'generic'
        if "reddit.com" in url: url_type = 'reddit'
        elif "backrooms" in url: url_type = 'backrooms'
        prompt = generate_dynamic_prompt(url_type, scraped_content, channel_name, narrator_style, is_short_form)
        script_data = generate_script_from_prompt(api_key, prompt)

        if not script_data:
            update_status(job_id, "Error: AI failed to generate script.", error=True)
            raise Exception("AI failed to generate script.")
        
        scenes_with_assets = []
        voice_settings = job_data.get('voice_settings', {})
        speaker_map = {}

        for i, scene_script in enumerate(script_data):
            update_status(job_id, f"Generating assets for scene {i+1} of {len(script_data)}...")
            speaker_key = scene_script.get('speaker', 'narrator').lower()
            if speaker_key.startswith('commenter'):
                if speaker_key not in speaker_map:
                    speaker_map[speaker_key] = f'commenter{len(speaker_map) + 1}'
            mapped_speaker = speaker_map.get(speaker_key, speaker_key)
            voice_name = voice_settings.get(mapped_speaker, voice_settings.get('narrator'))
            
            image_path = os.path.join(input_dir, f"scene_{i}.png")
            audio_path = os.path.join(input_dir, f"scene_{i}.mp3")

            generate_image_with_retries(api_key, scene_script.get('image_prompt'), image_path, is_short_form)
            duration = generate_audio(api_key, scene_script.get('line'), voice_name, audio_path)
            
            image_url = upload_to_s3(image_path, f"jobs/{job_id}/input/scene_{i}.png")
            audio_url = upload_to_s3(audio_path, f"jobs/{job_id}/input/scene_{i}.mp3")
            
            scenes_with_assets.append({
                'duration': duration, 'image_url': image_url, 'audio_url': audio_url,
                'local_image_path': image_path, 'local_audio_path': audio_path
            })
        
        update_status(job_id, "All assets generated, preparing for video render...")
        
        local_bg_music_path = None
        if job_data.get('background_music_url'):
            bg_music_key = urlparse(job_data['background_music_url']).path.lstrip('/')
            local_bg_music_path = os.path.join(input_dir, "bg_music.mp3")
            s3.download_file(AWS_S3_BUCKET_NAME, bg_music_key, local_bg_music_path)

        update_status(job_id, "Building video clips...")
        intermediate_video_paths = []
        caption_settings = job_data.get('caption_settings', {})
        framerate = 24

        for i, scene in enumerate(scenes_with_assets):
            duration = scene.get('duration', 1.0)
            intermediate_path = os.path.join(output_dir, f"scene_{i}.mp4")
            
            fade_duration = 0.5
            total_frames = int(duration * framerate)

            if is_short_form:
                # Vertical video (9:16), crop and pad
                filter_complex = (
                    f"[0:v]scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,setsar=1[vbase];"
                    f"[vbase]zoompan=z='min(zoom+0.0005,1.1)':d={total_frames}:s=720x1280[vzoomed];"
                    f"[vzoomed]fade=in:st=0:d={fade_duration},fade=out:st={duration - fade_duration}:d={fade_duration}[v{i}]"
                )
            else:
                # Horizontal video (16:9)
                filter_complex = (
                    f"[0:v]scale=1280:720,setsar=1[vbase];"
                    f"[vbase]zoompan=z='min(zoom+0.0005,1.1)':d={total_frames}:s=1280x720[vzoomed];"
                    f"[vzoomed]fade=in:st=0:d={fade_duration},fade=out:st={duration - fade_duration}:d={fade_duration}[v{i}]"
                )

            ffmpeg_scene_cmd = [
                FFMPEG_PATH, '-y', '-loop', '1', '-r', str(framerate), '-i', scene['local_image_path'],
                '-i', scene['local_audio_path'], '-filter_complex', filter_complex,
                '-map', f'[v{i}]', '-map', '1:a', '-c:v', 'libx264', '-preset', 'fast',
                '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-t', str(duration), intermediate_path
            ]
            subprocess.run(ffmpeg_scene_cmd, check=True, capture_output=True, text=True)
            intermediate_video_paths.append(intermediate_path)

        update_status(job_id, "Stitching video clips together...")
        concat_list_path = os.path.join(output_dir, "concat_list.txt")
        with open(concat_list_path, 'w') as f:
            for path in intermediate_video_paths:
                f.write(f"file '{os.path.basename(path)}'\n")

        video_no_music_path = os.path.join(output_dir, "final_no_music.mp4")
        ffmpeg_concat_cmd = [FFMPEG_PATH, '-y', '-f', 'concat', '-safe', '0', '-i', concat_list_path, '-c', 'copy', video_no_music_path]
        subprocess.run(ffmpeg_concat_cmd, check=True, capture_output=True, text=True)

        final_video_path = os.path.join(output_dir, "final_video.mp4")
        if local_bg_music_path:
            update_status(job_id, "Adding background music...")
            music_volume = int(caption_settings.get('musicVolume', 13)) / 100.0
            mix_filter = f"[1:a]volume={music_volume}[bga];[0:a][bga]amix=inputs=2:duration=first[a]"
            ffmpeg_mix_cmd = [
                FFMPEG_PATH, '-y', '-i', video_no_music_path, '-i', local_bg_music_path,
                '-filter_complex', mix_filter, '-map', '0:v', '-map', '[a]',
                '-c:v', 'copy', '-c:a', 'aac', '-shortest', final_video_path
            ]
            subprocess.run(ffmpeg_mix_cmd, check=True, capture_output=True, text=True)
        else:
            os.rename(video_no_music_path, final_video_path)
            
        update_status(job_id, "Finalizing and uploading video...")
        final_video_key = f"jobs/{job_id}/output/final_video.mp4"
        upload_to_s3(final_video_path, final_video_key)

        if preview_code:
            call_update_code_usage(preview_code)
            
        print(f"✅ Job {job_id} complete! Final video uploaded.")

    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        try:
            s3_client_check = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
            status_key = f"jobs/{job_id}/status.json"
            status_obj = s3_client_check.get_object(Bucket=AWS_S3_BUCKET_NAME, Key=status_key)
            status_data = json.loads(status_obj['Body'].read())
            if status_data.get('status') != 'failed':
                update_status(job_id, error_message, error=True)
        except Exception:
            update_status(job_id, error_message, error=True)
        print(f"❌ ERROR processing job {job_id}: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            print("--- FFMPEG STDERR ---")
            print(e.stderr)
        sys.exit(1)
    finally:
        if os.path.exists(input_dir): shutil.rmtree(input_dir)
        if os.path.exists(output_dir): shutil.rmtree(output_dir)
        print(f"Cleaned up local files for job {job_id}.")

# --- Main Execution Block ---
if __name__ == "__main__":
    if not JOB_ID:
        print("❌ ERROR: JOB_ID environment variable not set.")
        sys.exit(1)
    
    s3_main = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    job_key = f"jobs/{JOB_ID}/job.json"
    local_job_path = f"/tmp/{JOB_ID}.json"
    
    try:
        s3_main.download_file(AWS_S3_BUCKET_NAME, job_key, local_job_path)
        with open(local_job_path) as f:
            job_details = json.load(f)
        process_job(job_details)
    except Exception as e:
        print(f"❌ ERROR: Failed to fetch or run job. Error: {e}")
        update_status(JOB_ID, f"Failed to start job: {e}", error=True)
        sys.exit(1)

