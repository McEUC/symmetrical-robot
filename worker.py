import os
import sys
import json
import subprocess
import time
import boto3
from urllib.parse import urlparse
import shutil

# --- Part 1: Configuration ---
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET_NAME = os.environ.get('AWS_S3_BUCKET_NAME')
JOB_ID = os.environ.get('JOB_ID')
FFMPEG_PATH = "ffmpeg"

# --- Part 2: The Core Video Processing Function ---
def process_job(job_data):
    job_id = job_data.get('job_id', 'unknown-job')
    print(f"\n🚀 Starting job: {job_id}")

    input_dir = f"./{job_id}_input"
    output_dir = f"./{job_id}_output"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    
    try:
        print("Downloading assets...")
        scenes = job_data.get('scenes', [])
        for scene in scenes:
            image_key = urlparse(scene.get('image_url', '')).path.lstrip('/')
            audio_key = urlparse(scene.get('audio_url', '')).path.lstrip('/')
            scene['local_image_path'] = os.path.join(input_dir, os.path.basename(image_key))
            scene['local_audio_path'] = os.path.join(input_dir, os.path.basename(audio_key))
            s3.download_file(AWS_S3_BUCKET_NAME, image_key, scene['local_image_path'])
            s3.download_file(AWS_S3_BUCKET_NAME, audio_key, scene['local_audio_path'])
        
        local_bg_music_path = None
        if job_data.get('background_music_url'):
            bg_music_key = urlparse(job_data.get('background_music_url')).path.lstrip('/')
            local_bg_music_path = os.path.join(input_dir, "bg_music.mp3")
            s3.download_file(AWS_S3_BUCKET_NAME, bg_music_key, local_bg_music_path)
        print("Assets downloaded.")

        print("Building and executing FFmpeg command (CPU mode)...")
        inputs, filter_chains, video_concat_streams, audio_concat_streams = [], [], [], []
        framerate = 24
        caption_settings = job_data.get('caption_settings', {})

        for i, scene in enumerate(scenes):
            duration = scene.get('duration', 1.0)
            inputs.extend(['-loop', '1', '-i', scene['local_image_path']])
            inputs.extend(['-i', scene['local_audio_path']])
            video_concat_streams.append(f"[v{i}]")
            audio_concat_streams.append(f"[a{i}]")

            dialogue_text = scene.get('line', '')
            safe_caption = dialogue_text.replace("'", r"’").replace(':', r'\:').replace('%', r'%%').replace(',', r'\,')
            fade_duration = 0.5
            total_frames = int(duration * framerate)
            y_pos_map = {"bottom": "(h-text_h)-20", "middle": "(h-text_h)/2", "top": "20"}
            
            y_pos = y_pos_map.get(caption_settings.get('position', 'bottom'), "(h-text_h)-20")
            font_size = caption_settings.get('size', 35)
            font_color = caption_settings.get('color', '#FFFFFF')
            font_file = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"

            # --- THIS IS THE CORRECTED FILTER CHAIN ---
            # Added a 'trim' filter to explicitly set the duration of the video segment.
            filter_chains.extend([
                f"[{i*2}:v]trim=duration={duration},setpts=PTS-STARTPTS[v{i}_trimmed]",
                f"[v{i}_trimmed]scale=1280:720,zoompan=z='min(zoom+0.001,1.2)':d={total_frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1280x720[v{i}_zoomed]",
                f"[v{i}_zoomed]fade=in:st=0:d={fade_duration},fade=out:st={duration - fade_duration}:d={fade_duration}[v{i}_faded]",
                f"[v{i}_faded]drawtext=fontfile='{font_file}':text='{safe_caption}':fontsize={font_size}:fontcolor={font_color}:x=(w-tw)/2:y={y_pos}:box=1:boxcolor=black@0.5:boxborderw=5[v{i}]",
                f"[{i*2+1}:a]asetpts=PTS-STARTPTS[a{i}]"
            ])
            # --- END OF CORRECTION ---

        filter_chains.append(f"{''.join(video_concat_streams)}concat=n={len(scenes)}:v=1[outv]")
        filter_chains.append(f"{''.join(audio_concat_streams)}concat=n={len(scenes)}:v=0:a=1[maina]")
        
        final_video_path = os.path.join(output_dir, "final_video.mp4")
        ffmpeg_cmd = [FFMPEG_PATH, '-y', *inputs]

        if local_bg_music_path:
            ffmpeg_cmd.extend(['-i', local_bg_music_path])
            bg_music_index = len(scenes) * 2
            filter_chains.append(f"[{bg_music_index}:a]volume=0.2[bga]")
            filter_chains.append(f"[maina][bga]amix=inputs=2:duration=first[outa]")
            map_audio = '[outa]'
        else:
            map_audio = '[maina]'
            
        filter_complex_str = ";".join(filter_chains)
        
        ffmpeg_cmd.extend(['-filter_complex', filter_complex_str, '-map', '[outv]', '-map', map_audio, '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', final_video_path])

        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print("FFmpeg finished.")

        final_video_key = f"jobs/{job_id}/output/final_video.mp4"
        s3.upload_file(final_video_key, AWS_S3_BUCKET_NAME, final_video_key)
        print(f"✅ Job {job_id} complete! Final video uploaded.")

        print("Cleaning up temporary S3 files...")
        s3_input_prefix = f"jobs/{job_id}/input/"
        response = s3.list_objects_v2(Bucket=AWS_S3_BUCKET_NAME, Prefix=s3_input_prefix)
        if 'Contents' in response:
            objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
            s3.delete_objects(Bucket=AWS_S3_BUCKET_NAME, Delete={'Objects': objects_to_delete})
            print(f"Deleted {len(objects_to_delete)} temporary files from S3.")
        
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

# --- Part 4: Main Execution Block ---
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
