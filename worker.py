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
from pexels_api import API # New import for Pexels

# --- Configuration ---
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET_NAME = os.environ.get('AWS_S3_BUCKET_NAME')
JOB_ID = os.environ.get('JOB_ID')
FFMPEG_PATH = "ffmpeg"
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
PEXELS_API_KEY = os.environ.get('PEXELS_API_KEY') # New Pexels API Key
FLASK_APP_URL = f"http://{os.environ.get('FLASK_VM_IP')}:5000"

IMAGE_MODELS_TO_TRY = [
    "imagen-4.0-fast-generate-preview-06-06",
    "imagen-3.0-fast-generate-001",
    "imagen-3.0-generate-002",
    "imagegeneration@006"
]

# --- Status Reporting Function ---
def update_status(job_id, message, error=False):
    """Creates or updates a status.json file in S3 for the job."""
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    status_data = { "status": "failed" if error else "in_progress", "message": message, "timestamp": time.time() }
    s3_client.put_object( Bucket=AWS_S3_BUCKET_NAME, Key=f"jobs/{job_id}/status.json", Body=json.dumps(status_data), ContentType='application/json' )
    print(f"Status update: {message}")

# --- NEW: Stock Footage Function ---
def get_stock_footage(api_key, query, output_path, is_short_form=False):
    """Searches for and downloads the best stock video clip from Pexels."""
    print(f"Searching for stock footage with query: '{query}'")
    try:
        api = API(api_key)
        orientation = "portrait" if is_short_form else "landscape"
        api.search_videos(query, page=1, results_per_page=5, orientation=orientation)
        videos = api.get_entries()
        if not videos:
            print(f"No stock footage found for '{query}'.")
            return None

        # Find the best quality video file (usually the highest resolution)
        best_video = videos[0]
        best_file = max(best_video.video_files, key=lambda f: f.width)
        
        # Download the video
        video_response = requests.get(best_file.link, stream=True)
        video_response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in video_response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded stock footage to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error fetching stock footage: {e}")
        return None


# --- HELPER FUNCTIONS (Modified and existing) ---
def upload_to_s3(file_path, object_name):
    """Uploads a file to an S3 bucket."""
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    s3_client.upload_file(file_path, AWS_S3_BUCKET_NAME, object_name)
    print(f"Uploaded {os.path.basename(file_path)} to S3.")

def generate_script_from_prompt(api_key, prompt):
    """Calls the Gemini API to generate the video script."""
    # ... (This function remains unchanged)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={api_key}"
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
    # ... (This function remains unchanged)
    print(f"Generating image for prompt: '{prompt}'")
    aspect_ratio = "9:16" if is_short_form else "16:9"
    
    models_to_attempt = IMAGE_MODELS_TO_TRY * 2

    for model_name in models_to_attempt:
        try:
            print(f"Attempting image generation with model: {model_name}")
            vertexai.init(project=GCP_PROJECT_ID)
            model = ImageGenerationModel.from_pretrained(model_name)
            images = model.generate_images(
                prompt=prompt, number_of_images=1, aspect_ratio=aspect_ratio,
                negative_prompt="text, letters, words, watermark, signature, logo" )
            if images:
                images[0].save(location=output_path, include_generation_parameters=False)
                print(f"Successfully generated image with {model_name}.")
                return
            else:
                print(f"Model {model_name} returned no images.")
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
    
    raise Exception(f"Image generation failed for prompt '{prompt}' after trying all models.")


def generate_audio(api_key, text, voice_name, output_path):
    # ... (This function remains unchanged)
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


def process_url_content(url):
    # ... (This function remains unchanged)
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
    # ... (This function remains unchanged)
    try:
        update_url = f"{FLASK_APP_URL}/update-code-usage"
        response = requests.post(update_url, json={"code": code}, timeout=10)
        response.raise_for_status()
        print(f"Successfully updated usage for code: {code}")
    except Exception as e:
        print(f"WARNING: Failed to update code usage for '{code}'. Error: {e}")


def generate_dynamic_prompt(content, channel_name, narrator_style, is_short_form):
    # ... (This function remains unchanged, simplified slightly)
    style_instruction = f"- **Host Persona:** You are a narrator with a '{narrator_style}' style." if narrator_style else "- **Host Persona:** You are a neutral, informative narrator."
    branded_intro = f'Start with the line: "Welcome back to {channel_name}!"' if channel_name else "Start with a compelling hook (1-2 sentences)."
    branded_outro = f'End with: "Thanks for watching {channel_name}! Subscribe for more!"' if channel_name else "End with a concluding thought."
    scene_count = "8 to 12 scenes" if is_short_form else "15 to 20 scenes"
    image_style = "'A cinematic, hyper-realistic digital painting with dramatic lighting.'"

    prompt = f"""
ACT as a YouTube host. Create a script based on the provided content for a video with {scene_count}.
**Style:**
{style_instruction}
- Add original commentary and offer a strong opinion.
**INPUT:**
- **Web Content:** [{content}]
**TASK: Structure the script EXACTLY as follows in a valid JSON object:**
{{
  "script": {{
    "title": "A catchy, viral-style title for the video",
    "scenes": [
      {{ 
        "line": "The spoken line for this scene...", 
        "image_prompt": "A detailed visual description for an AI image generator. Style: {image_style}. No text.",
        "video_search_query": "A concise, 2-3 word search query for stock footage (e.g., 'data center', 'forest path')."
      }}
    ]
  }}
}}
"""
    return prompt

# --- Core Video Processing Function (MODIFIED) ---
def process_job(job_data):
    job_id = job_data.get('job_id')
    print(f"\n🚀 Starting job: {job_id}")
    update_status(job_id, "Worker started, preparing assets...")

    input_dir = f"./{job_id}_input"
    output_dir = f"./{job_id}_output"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get job data
        api_key = job_data.get('api_key')
        url = job_data.get('url')
        preview_code = job_data.get('preview_code')
        channel_name = job_data.get('channel_name')
        narrator_style = job_data.get('narrator_style')
        is_short_form = job_data.get('is_short_form', False)
        visual_source = job_data.get('visual_source', 'ai_image') # NEW: Get visual source

        # Script generation
        update_status(job_id, "Scraping the URL...")
        scraped_content = process_url_content(url)
        update_status(job_id, "Generating script with AI...")
        prompt = generate_dynamic_prompt(scraped_content, channel_name, narrator_style, is_short_form)
        script_data = generate_script_from_prompt(api_key, prompt)
        if not script_data: raise Exception("AI failed to generate script.")
        
        scenes_with_assets = []
        voice_settings = job_data.get('voice_settings', {})

        # Asset generation loop (MODIFIED)
        for i, scene_script in enumerate(script_data):
            update_status(job_id, f"Generating assets for scene {i+1} of {len(script_data)}...")
            
            # Visual asset generation (MODIFIED)
            visual_path = None
            if visual_source == 'stock_video':
                video_search_query = scene_script.get('video_search_query', 'abstract')
                visual_path = get_stock_footage(PEXELS_API_KEY, video_search_query, os.path.join(input_dir, f"scene_{i}.mp4"), is_short_form)
            
            # Fallback to AI image if stock video fails or is not selected
            if not visual_path:
                visual_path = os.path.join(input_dir, f"scene_{i}.png")
                generate_image_with_retries(api_key, scene_script.get('image_prompt'), visual_path, is_short_form)

            # Audio asset generation
            audio_path = os.path.join(input_dir, f"scene_{i}.mp3")
            voice_name = voice_settings.get('narrator', 'en-US-Studio-Q') # Simplified for now
            duration = generate_audio(api_key, scene_script.get('line'), voice_name, audio_path)
            
            scenes_with_assets.append({
                'duration': duration, 
                'local_visual_path': visual_path, 
                'local_audio_path': audio_path
            })
        
        update_status(job_id, "All assets generated, preparing for video render...")
        
        # Video rendering loop (MODIFIED)
        intermediate_video_paths = []
        for i, scene in enumerate(scenes_with_assets):
            duration = scene['duration']
            intermediate_path = os.path.join(output_dir, f"scene_{i}.mp4")
            local_visual_path = scene['local_visual_path']
            is_video_input = local_visual_path.endswith('.mp4')

            resolution = "720:1280" if is_short_form else "1280:720"
            
            filter_complex = []
            if is_video_input:
                # Input is a video file
                filter_complex.append(f"[0:v]scale={resolution}:force_original_aspect_ratio=increase,crop={resolution},setsar=1[v]")
            else:
                # Input is an image file
                total_frames = int(duration * 24)
                filter_complex.append(f"[0:v]scale={resolution},setsar=1[vbase]")
                filter_complex.append(f"[vbase]zoompan=z='min(zoom+0.0005,1.1)':d={total_frames}:s={resolution.replace(':', 'x')}[v]")

            ffmpeg_scene_cmd = [
                FFMPEG_PATH, '-y', 
                '-i', local_visual_path,
                '-i', scene['local_audio_path'],
                '-filter_complex', ";".join(filter_complex),
                '-map', '[v]', '-map', '1:a',
                '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-shortest', # Use shortest to trim video to audio length
                '-t', str(duration), intermediate_path
            ]
            if not is_video_input: ffmpeg_scene_cmd.insert(3, '-loop') and ffmpeg_scene_cmd.insert(4, '1')

            subprocess.run(ffmpeg_scene_cmd, check=True, capture_output=True, text=True)
            intermediate_video_paths.append(intermediate_path)
        
        # Final video stitching
        update_status(job_id, "Stitching video clips together...")
        concat_list_path = os.path.join(output_dir, "concat_list.txt")
        with open(concat_list_path, 'w') as f:
            for path in intermediate_video_paths:
                f.write(f"file '{os.path.basename(path)}'\n")
        
        final_video_path = os.path.join(output_dir, "final_video.mp4")
        ffmpeg_concat_cmd = [FFMPEG_PATH, '-y', '-f', 'concat', '-safe', '0', '-i', concat_list_path, '-c', 'copy', final_video_path]
        subprocess.run(ffmpeg_concat_cmd, check=True, capture_output=True, text=True)
            
        # Upload and finalize
        update_status(job_id, "Finalizing and uploading video...")
        upload_to_s3(final_video_path, f"jobs/{job_id}/output/final_video.mp4")
        if preview_code: call_update_code_usage(preview_code)
            
        print(f"✅ Job {job_id} complete! Final video uploaded.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_message = f"An unexpected error occurred: {e}"
        update_status(job_id, error_message, error=True)
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





