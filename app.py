import os
import uuid
import json
import requests
import boto3
from flask import Flask, request, jsonify, render_template

# --- CONFIGURATION ---
app = Flask(__name__, static_folder='static', template_folder='templates')
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- SECRETS & CONFIG (from environment) ---
GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME")
GITHUB_REPO_NAME = os.environ.get("GITHUB_REPO_NAME")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME")
AWS_S3_REGION = os.environ.get("AWS_S3_REGION", "us-east-1")

def upload_to_s3(file_path, object_name):
    """Uploads a file to an S3 bucket."""
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_S3_REGION)
    try:
        s3_client.upload_file(file_path, AWS_S3_BUCKET_NAME, object_name)
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-video', methods=['POST'])
def handle_video_generation():
    try:
        form_data = request.form
        job_id = str(uuid.uuid4())
        job_folder = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
        os.makedirs(job_folder, exist_ok=True)

        # Create a job definition file with all necessary info for the worker
        job_data = {
            "job_id": job_id,
            "url": form_data.get('url'),
            "api_key": form_data.get('apiKey'),
            "channel_name": form_data.get('channelName'),
            "narrator_style": form_data.get('narratorStyle'),
            "is_short_form": form_data.get('isShortForm') == 'true',
            "voice_settings": json.loads(form_data.get('voiceSettings')),
            "caption_settings": json.loads(form_data.get('captionSettings')),
        }
        
        bg_music_url = None
        if 'backgroundMusic' in request.files and request.files['backgroundMusic'].filename != '':
            music_file = request.files['backgroundMusic']
            bg_music_path = os.path.join(job_folder, "bg_music.mp3")
            music_file.save(bg_music_path)
            music_s3_key = f"jobs/{job_id}/input/bg_music.mp3"
            upload_to_s3(bg_music_path, music_s3_key)
            bg_music_url = f"https://{AWS_S3_BUCKET_NAME}.s3.amazonaws.com/{music_s3_key}"

        job_data["background_music_url"] = bg_music_url

        job_file_path = os.path.join(job_folder, 'job.json')
        with open(job_file_path, 'w') as f:
            json.dump(job_data, f)
        upload_to_s3(job_file_path, f"jobs/{job_id}/job.json")

        print(f"Triggering GitHub Action for job: {job_id}")
        headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {GITHUB_TOKEN}", "X-GitHub-Api-Version": "2022-11-28"}
        data = {"event_type": "video-job", "client_payload": {"job_id": job_id}}
        dispatch_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO_NAME}/dispatches"
        api_response = requests.post(dispatch_url, json=data, headers=headers, timeout=10)
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
    except s3_client.exceptions.ClientError:
        pass

    try:
        status_object_key = f"jobs/{job_id}/status.json"
        status_obj = s3_client.get_object(Bucket=AWS_S3_BUCKET_NAME, Key=status_object_key)
        status_data = json.loads(status_obj['Body'].read().decode('utf-8'))
        return jsonify(status_data)
    except s3_client.exceptions.ClientError:
        return jsonify({"status": "pending", "message": "Job is queued and waiting for the worker..."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
