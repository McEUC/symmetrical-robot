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
            response = requests.post(endpoint, headers=headers, json=data, timeout=60)
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

def generate_image(api_key, prompt, output_path):
    """Generates an image using Vertex AI, with a self-healing retry mechanism."""
    print(f"Generating real image for prompt: '{prompt}'")
    max_retries = 3
    current_prompt = prompt
    for attempt in range(max_retries):
        try:
            vertexai.init()
            model = ImageGenerationModel.from_pretrained("imagegeneration@006")
            images = model.generate_images(prompt=current_prompt, number_of_images=1, aspect_ratio="16:9", negative_prompt="text, letters, words, watermark, signature, logo")
            if images:
                images[0].save(location=output_path, include_generation_parameters=False)
                print(f"Successfully saved image to {output_path}")
                return
            raise Exception("API returned an empty list of images.")
        except Exception as e:
            print(f"Imagen API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("Attempting to rewrite prompt to be safer...")
                try:
                    fixer_prompt = f"""The following prompt for an AI image generator was rejected, likely for violating a safety policy: "{current_prompt}"
Rewrite the prompt to be safer while preserving the original's core artistic and atmospheric intent. Focus on removing potentially sensitive words and describe the scene more abstractly if needed. Do not describe graphic violence, gore, or self-harm.
Respond ONLY with the new prompt text."""
                    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
                    headers = {"Content-Type": "application/json"}
                    data = {"contents": [{"parts": [{"text": fixer_prompt}]}]}
                    response = requests.post(endpoint, headers=headers, json=data, timeout=45)
                    response.raise_for_status()
                    response_json = response.json()
                    if response_json.get('candidates'):
                        new_prompt = response_json['candidates'][0]['content']['parts'][0]['text'].strip()
                        print(f"Generated new, safer prompt: '{new_prompt}'")
                        current_prompt = new_prompt
                except Exception as fixer_e:
                    print(f"Failed to rewrite the prompt: {fixer_e}")
                print("Waiting 60 seconds before retrying...")
                time.sleep(60)
            else:
                raise e
    raise Exception(f"Image generation failed for prompt '{prompt}' after multiple attempts.")

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

# --- URL PROCESSING FUNCTIONS ---
def process_reddit_thread(url, api_key):
    """Scrapes a Reddit thread by fetching its HTML and extracting text."""
    print(f"Processing Reddit URL using HTML scraping: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        page = requests.get(url, headers=headers, timeout=15)
        page.raise_for_status()
        soup = BeautifulSoup(page.content, 'html.parser')
        
        title_tag = soup.find('h1') or soup.find('h2') or soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else 'No Title Found'
        
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        post_text = soup.get_text(separator=' ', strip=True)
        
        full_text = f"Title: {title}. Post Content: {post_text}"
        print(f"Extracted Reddit content: {full_text[:300]}...")
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch the Reddit URL: {e}")

    prompt = f"""
"ACT as a witty, opinionated, and funny YouTube host for the channel 'Tales from the Upboat'. Your task is to create an engaging video script based on a Reddit post.
**Channel Style:**
- **Host Persona:** You are sharp, humorous, and not afraid to share your personal take.
- **Content Style:** The video should feel like a conversation. It's not just reading; it's reacting, analyzing, and adding significant original commentary.
**INPUT:**
- **Reddit Content (Title and Post):** [{full_text}]
**TASK: Structure the script EXACTLY as follows:**
1.  **Branded Intro (Speaker: narrator):** Start with the line: "Welcome back to Tales from the Upboat, where we dive headfirst into the best stories the internet has to offer."
2.  **Hook (Speaker: narrator):** Create a compelling, short hook (1-2 sentences) that teases the main story's theme or conflict based on the title.
3.  **Main Story Reading (Speaker: narrator):** Read the Reddit post content. **CRITICAL: You MUST break the main story reading into multiple script segments, each about 3-5 sentences long. Each of these segments must have its own unique `image_prompt` that describes a distinct visual scene from that specific part of the story.**
4.  **Extensive Post-Story Commentary (Speaker: commenter1):** After reading the post, provide a detailed, funny, and insightful commentary. This should be a significant portion of the script. Analyze the situation, share a personal anecdote, and offer a strong opinion.
5.  **Branded Outro (Speaker: narrator):** End the video with the exact lines: "And that's all the time we have for today on Tales from the Upboat. If you liked this story, be sure to hit that subscribe button and ring the bell so you don't miss the next one. Until next time, stay curious and keep scrolling."
**IMAGE PROMPT RULES (Apply to ALL prompts):**
- **Consistent Style:** Every prompt MUST include these keywords: 'A vibrant and detailed digital illustration in a classic Japanese anime style.' The image_prompt field must always contain a string describing the scene and cannot be null.
- **No Text:** Prompts MUST NOT contain any words, text, or letters. The final image should be purely visual.
**OUTPUT FORMAT:**
Return a valid JSON object strictly following this schema:
{{
  "script": {{
    "title": "A catchy, viral-style title for the video based on the post",
    "scenes": [
      {{
        "speaker": "narrator" | "commenter1",
        "line": "The lines to be spoken.",
        "image_prompt": "A highly detailed, cinematic prompt for an image, following all rules above."
      }}
    ]
  }}
}}
"""
    return generate_script_from_prompt(api_key, prompt)

def process_generic_url(url, api_key):
    """Scrapes a generic webpage and generates a documentary-style script."""
    print(f"Processing generic URL: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        page = requests.get(url, headers=headers, timeout=15)
        page.raise_for_status()
        soup = BeautifulSoup(page.content, 'html.parser')

        page_title_tag = soup.find('h1') or soup.find('title')
        page_title = page_title_tag.get_text(strip=True) if page_title_tag else url
        
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        page_text = soup.get_text(separator=' ', strip=True)[:8000]
        if not page_text:
            raise Exception("Could not extract any text from the URL.")
        print(f"Extracted title: {page_title} and {len(page_text)} characters of text.")
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch the URL: {e}")

    prompt = f"""
ACT as the host of 'Web Weaver,' a YouTube channel that transforms articles and web content into compelling visual stories.
**Channel Style:**
- **Host Persona:** You are a thoughtful and engaging documentarian. Your tone is clear, informative, and slightly dramatic.
- **Content Style:** Synthesize, paraphrase, and dramatize the provided text. Weave the key facts into a cohesive narrative.
**INPUT:**
- **Article Title:** [{page_title}]
- **Article Content:** [{page_text}]
**TASK: Structure the script EXACTLY as follows:**
1.  **Script Length:** Generate between 10 and 15 scenes.
2.  **Branded Intro (Speaker: narrator):** Start with: "In the vast network of information that is our world, some stories are waiting in plain sight. Welcome to Web Weaver, where we unravel the code of content to bring you the story within."
3.  **Hook (Speaker: narrator):** Create an engaging hook (2-3 sentences) that introduces '{page_title}'.
4.  **Content Weaving (Multiple Scenes):**
    -   **Information Synthesis (Speaker: narrator):** Rephrase and narrate the key information from the article in your own documentary style.
    -   **Insightful Analysis (Speaker: commenter1):** After presenting a key piece of information, provide analysis that explains its importance or context.
5.  **Branded Outro (Speaker: narrator):** End with: "The story doesn't end here; it's just one thread in a much larger tapestry. Thanks for joining us on Web Weaver. Stay curious, and keep exploring."
**IMAGE PROMPT RULES (Apply to ALL prompts):**
-   **Consistent Style:** Every prompt MUST include the phrase: 'A cinematic, hyper-realistic digital painting with dramatic lighting'.
-   **No Text:** Prompts MUST NOT contain any words, text, or letters.
Respond ONLY with a valid JSON object in the format: {{"scenes": [{{"speaker": "narrator", "line": "Dialogue.", "image_prompt": "Image prompt."}}]}}
"""
    return generate_script_from_prompt(api_key, prompt)

def process_backrooms_wiki(url, api_key):
    """Scrapes a Backrooms wiki page and generates a script."""
    print("Processing Backrooms Wiki URL...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        page = requests.get(url, headers=headers, timeout=10)
        page.raise_for_status()
        soup = BeautifulSoup(page.content, 'html.parser')
        article_title = soup.title.string.replace(" Wiki | Fandom", "").strip() if soup.title and soup.title.string else "Backrooms Level"
        
        content_div = soup.find('div', class_='mw-parser-output')
        if not content_div:
            raise Exception("Could not find the main content area ('mw-parser-output') on the wiki page.")
        wiki_text = content_div.get_text(separator=' ', strip=True)[:6000]
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch the Backrooms Wiki URL: {e}")
    
    prompt = f"""
ACT as the host of 'Liminal Echoes,' a YouTube channel that explores the unsettling depths of the Backrooms. Your task is to transform a dry wiki article into a chilling, narrative-driven video script.
**Channel Style:**- **Host Persona:** You are a haunted storyteller. Your tone is hushed, conspiratorial, and filled with a sense of dread. You speak as if you are sharing forbidden knowledge that has taken a toll on you. You are not a simple narrator; you are an interpreter of the abyss.- **Content Style:** The video MUST NOT read the wiki word-for-word. Instead, you will synthesize, paraphrase, and dramatize the information. Weave the facts from the wiki into a creepy, first-person or second-person narrative. The script must be dominated by your original, unsettling commentary which should connect different pieces of information on the page to build a cohesive sense of horror.
**INPUT:**- **Wiki Article Title:** [{article_title}]- **Wiki Article Content:** [{wiki_text}]
**TASK: Structure the script EXACTLY as follows:**
1.  **Script Length:** The final script should be substantial. Generate between 12 and 18 scenes to create a video with a total runtime between 5 and 8 minutes.
2.  **Branded Intro (Speaker: narrator):** Start with the exact line: "Listen closely. Can you hear it? That hum in the static between worlds? Welcome back to Liminal Echoes, where we give voice to the silence of places that shouldn't exist."
3.  **Hook (Speaker: narrator):** Create a deeply unsettling hook (2-3 sentences) that introduces '{article_title}' by asking a disturbing question or painting a chilling mental image for the viewer.
4.  **Content Weaving Loop:** Your primary task is to process the wiki article section by section (e.g., Description, Entities, Entrances, Exits). For each section, create a two-part sequence:    -   **Information Synthesis (Speaker: narrator):** **DO NOT READ THE WIKI VERBATIM.** You must rephrase and narrate the key information from the wiki section in your own creepy style. Describe it as if you are seeing it, or as if the viewer is the one trapped there. For example, instead of 'The walls are yellow,' say 'An endless, sickening yellow stains the walls, the color of old bruises and decay...'    -   **Unsettling Analysis (Speaker: commenter1):** Immediately after synthesizing a piece of information, provide extensive, creepy, and speculative analysis. This is the heart of the video. Question the 'why'. Speculate on the malevolent intelligence behind the architecture. Connect the description of the level to the entities found within it. Discuss the psychological toll it would take. Ask disturbing rhetorical questions that linger with the viewer.
5.  **Branded Outro (Speaker: narrator):** End the video with the exact lines: "The information stops here, but the feeling doesn't. Be wary of the quiet places and the patterns you start to see in the static. The echoes are always listening. Until next time, try not to get lost."
**IMAGE PROMPT RULES (Apply to ALL prompts):**-   **Consistent Style:** Every prompt MUST include ONE of these two style phrases: 'A grainy, unsettling, found-footage style photograph' OR 'A hyper-realistic, liminal space digital painting'. The image_prompt field must always contain a string describing the scene and cannot be null.-   **Atmosphere:** Prompts should focus on creating feelings of dread, isolation, and cosmic horror. Describe empty spaces, strange architecture, distorted figures in the distance, and analog-style visual artifacts.-   **No Text:** Prompts MUST NOT contain any words, text, or letters.
Respond ONLY with a valid JSON object in the format: {{"scenes": [{{"speaker": "narrator", "line": "Dialogue.", "image_prompt": "Image prompt."}}]}}
"""
    return generate_script_from_prompt(api_key, prompt)


# --- Core Video Processing Function ---
def process_job(job_data):
    job_id = job_data.get('job_id')
    print(f"\n🚀 Starting job: {job_id}")

    input_dir = f"./{job_id}_input"
    output_dir = f"./{job_id}_output"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    
    try:
        # --- PART 1: ASSET GENERATION ---
        print("🤖 Generating script and assets...")
        api_key = job_data.get('api_key')
        url = job_data.get('url')
        
        if "reddit.com" in url:
            script_data = process_reddit_thread(url, api_key)
        elif "backrooms" in url and ("wiki" in url or "fandom" in url):
            script_data = process_backrooms_wiki(url, api_key)
        else:
            script_data = process_generic_url(url, api_key)

        if not script_data:
            raise Exception("The AI failed to generate a valid script.")
        
        scenes_with_assets = []
        voice_settings = job_data.get('voice_settings', {})
        speaker_map = {}

        for i, scene_script in enumerate(script_data):
            print(f"  - Generating assets for scene {i+1}/{len(script_data)}...")
            
            speaker_key = scene_script.get('speaker', 'narrator').lower()
            if speaker_key.startswith('commenter'):
                if speaker_key not in speaker_map:
                    speaker_map[speaker_key] = f'commenter{len(speaker_map) + 1}'
            mapped_speaker = speaker_map.get(speaker_key, speaker_key)
            voice_name = voice_settings.get(mapped_speaker, voice_settings.get('narrator'))
            
            image_path = os.path.join(input_dir, f"scene_{i}.png")
            audio_path = os.path.join(input_dir, f"scene_{i}.mp3")

            generate_image(api_key, scene_script.get('image_prompt'), image_path)
            duration = generate_audio(api_key, scene_script.get('line'), voice_name, audio_path)
            
            image_url = upload_to_s3(image_path, f"jobs/{job_id}/input/scene_{i}.png")
            audio_url = upload_to_s3(audio_path, f"jobs/{job_id}/input/scene_{i}.mp3")
            
            scenes_with_assets.append({
                'duration': duration,
                'image_url': image_url,
                'audio_url': audio_url,
                'local_image_path': image_path,
                'local_audio_path': audio_path
            })
        print("✅ All assets generated and uploaded.")
        
        # --- PART 2: VIDEO RENDERING ---
        print("Downloading background music if available...")
        local_bg_music_path = None
        if job_data.get('background_music_url'):
            bg_music_key = urlparse(job_data['background_music_url']).path.lstrip('/')
            local_bg_music_path = os.path.join(input_dir, "bg_music.mp3")
            s3.download_file(AWS_S3_BUCKET_NAME, bg_music_key, local_bg_music_path)
            print("Background music downloaded.")

        print("Building individual scene videos...")
        intermediate_video_paths = []
        caption_settings = job_data.get('caption_settings', {})
        framerate = 24

        for i, scene in enumerate(scenes_with_assets):
            print(f"Processing scene {i+1}/{len(scenes_with_assets)}...")
            duration = scene.get('duration', 1.0)
            intermediate_path = os.path.join(output_dir, f"scene_{i}.mp4")
            
            fade_duration = 0.5
            total_frames = int(duration * framerate)

            filter_complex = (
                f"[0:v]trim=duration={duration},setpts=PTS-STARTPTS,scale=1280:720,setsar=1[vbase];"
                f"[vbase]zoompan=z='zoom+0.0005':d={total_frames}:s=1280x720[vzoomed];"
                f"[vzoomed]fade=in:st=0:d={fade_duration},fade=out:st={duration - fade_duration}:d={fade_duration}[v{i}]"
            )
            
            ffmpeg_scene_cmd = [
                FFMPEG_PATH, '-y',
                '-loop', '1', '-r', str(framerate), '-i', scene['local_image_path'],
                '-i', scene['local_audio_path'],
                '-filter_complex', filter_complex,
                '-map', f'[v{i}]',
                '-map', '1:a',
                '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv4z0p',
                '-c:a', 'aac', '-t', str(duration),
                intermediate_path
            ]
            subprocess.run(ffmpeg_scene_cmd, check=True, capture_output=True, text=True)
            intermediate_video_paths.append(intermediate_path)

        print("\nStitching scene videos together...")
        concat_list_path = os.path.join(output_dir, "concat_list.txt")
        with open(concat_list_path, 'w') as f:
            for path in intermediate_video_paths:
                f.write(f"file '{os.path.basename(path)}'\n")

        video_no_music_path = os.path.join(output_dir, "final_no_music.mp4")
        ffmpeg_concat_cmd = [FFMPEG_PATH, '-y', '-f', 'concat', '-safe', '0', '-i', concat_list_path, '-c', 'copy', video_no_music_path]
        subprocess.run(ffmpeg_concat_cmd, check=True, capture_output=True, text=True)

        final_video_path = os.path.join(output_dir, "final_video.mp4")
        if local_bg_music_path:
            print("Mixing in background music...")
            music_volume = int(caption_settings.get('musicVolume', 13)) / 100.0
            mix_filter = f"[1:a]volume={music_volume}[bga];[0:a][bga]amix=inputs=2:duration=first[a]"
            
            ffmpeg_mix_cmd = [
                FFMPEG_PATH, '-y',
                '-i', video_no_music_path,
                '-i', local_bg_music_path,
                '-filter_complex', mix_filter,
                '-map', '0:v', '-map', '[a]',
                '-c:v', 'copy', '-c:a', 'aac', '-shortest',
                final_video_path
            ]
            subprocess.run(ffmpeg_mix_cmd, check=True, capture_output=True, text=True)
        else:
            os.rename(video_no_music_path, final_video_path)
            
        print("FFmpeg finished.")
        final_video_key = f"jobs/{job_id}/output/final_video.mp4"
        upload_to_s3(final_video_path, final_video_key)
        print(f"✅ Job {job_id} complete! Final video uploaded.")

    except Exception as e:
        print(f"❌ ERROR processing job {job_id}: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            print("--- FFMPEG STDERR ---")
            print(e.stderr)
            print("--- END FFMPEG STDERR ---")
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
        print(f"Fetching job details from S3: {job_key}")
        s3_main.download_file(AWS_S3_BUCKET_NAME, job_key, local_job_path)
        with open(local_job_path) as f:
            job_details = json.load(f)
        
        process_job(job_details)
    except Exception as e:
        print(f"❌ ERROR: Failed to fetch or run job. Error: {e}")
        sys.exit(1)