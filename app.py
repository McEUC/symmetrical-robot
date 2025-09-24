import os
import uuid
import json
import requests
import boto3
import base64
from flask import Flask, request, jsonify, render_template, Response
from googleapiclient.discovery import build
import code_manager 

# --- CONFIGURATION ---
app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- SECRETS & CONFIG (from environment) ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") 
GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME")
GITHUB_REPO_NAME = os.environ.get("GITHUB_REPO_NAME")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME")
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY")

def upload_to_s3(file_path, object_name):
    """Uploads a file to an S3 bucket."""
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    try:
        s3_client.upload_file(file_path, AWS_S3_BUCKET_NAME, object_name)
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise

# --- FLASK ROUTES ---
@app.route('/')
def index():
    """Serves the landing page."""
    return render_template('landing.html')

@app.route('/generator')
def generator():
    """Serves the main generator application page."""
    return render_template('generator.html')

@app.route('/validate-code', methods=['POST'])
def validate_preview_code():
    """Endpoint for the frontend to validate a preview code."""
    data = request.get_json()
    code = data.get('code')
    if not code:
        return jsonify({"valid": False, "message": "Please enter a code."})
    return jsonify(code_manager.validate_code(code))

@app.route('/update-code-usage', methods=['POST'])
def update_code_usage_route():
    """Endpoint for the worker to call back to after a successful video generation."""
    data = request.get_json()
    code = data.get('code')
    result = code_manager.update_code_usage(code)
    if result["success"]:
        return jsonify(result), 200
    return jsonify(result), 400

@app.route('/preview-tts', methods=['POST'])
def preview_tts():
    """Generates a TTS audio sample for the frontend preview."""
    try:
        data = request.get_json()
        voice_name = data.get('voice')
        if not voice_name:
            return jsonify({"error": "Voice name is required."}), 400

        if not GOOGLE_API_KEY:
            return jsonify({"error": "Server TTS is not configured."}), 500

        preview_text = "This is a preview of the selected narrator's voice. You can adjust the background music volume to find the perfect balance for your video production."

        tts_service = build('texttospeech', 'v1', developerKey=GOOGLE_API_KEY)
        request_body = {
            'input': {'text': preview_text},
            'voice': {'languageCode': voice_name[:5], 'name': voice_name},
            'audioConfig': {'audioEncoding': 'MP3'}
        }
        response = tts_service.text().synthesize(body=request_body).execute()
        audio_content = base64.b64decode(response['audioContent'])
        
        return Response(audio_content, mimetype='audio/mpeg')

    except Exception as e:
        print(f"Error in TTS preview: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate-video', methods=['POST'])
def handle_video_generation():
    """Handles the main video generation job submission."""
    try:
        form_data = request.form
        preview_code = form_data.get('previewCode')

        if not preview_code or not code_manager.validate_code(preview_code)["valid"]:
            return jsonify({"error": "A valid preview code is required."}), 403
            
        if not GOOGLE_API_KEY or not PEXELS_API_KEY:
            return jsonify({"error": "Critical Server Error: API keys not configured."}), 500

        job_id = str(uuid.uuid4())
        job_folder = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
        os.makedirs(job_folder, exist_ok=True)

        job_data = {
            "job_id": job_id,
            "url": form_data.get('url'),
            "api_key": GOOGLE_API_KEY,
            "pexels_api_key": PEXELS_API_KEY,
            "preview_code": preview_code,
            "channel_name": form_data.get('channelName'),
            "narrator_style": form_data.get('narratorStyle'),
            "is_short_form": form_data.get('isShortForm') == 'true',
            "visual_source": form_data.get('visualSource'),
            "voice_settings": json.loads(form_data.get('voiceSettings')),
            "caption_settings": json.loads(form_data.get('captionSettings')),
            "background_music_url": None
        }
        
        # Handle background music upload
        if 'backgroundMusic' in request.files and request.files['backgroundMusic'].filename != '':
            music_file = request.files['backgroundMusic']
            music_filename = "bg_music" + os.path.splitext(music_file.filename)[1]
            bg_music_path = os.path.join(job_folder, music_filename)
            music_file.save(bg_music_path)
            music_s3_key = f"jobs/{job_id}/input/{music_filename}"
            upload_to_s3(bg_music_path, music_s3_key)
            job_data["background_music_url"] = f"https://{AWS_S3_BUCKET_NAME}.s3.amazonaws.com/{music_s3_key}"

        job_file_path = os.path.join(job_folder, 'job.json')
        with open(job_file_path, 'w') as f:
            json.dump(job_data, f)
        upload_to_s3(job_file_path, f"jobs/{job_id}/job.json")

        headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {GITHUB_TOKEN}"}
        data = {"event_type": "video-job", "client_payload": {"job_id": job_id}}
        dispatch_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO_NAME}/dispatches"
        requests.post(dispatch_url, json=data, headers=headers, timeout=10).raise_for_status()
        
        return jsonify({"message": "Job submitted successfully!", "jobId": job_id})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Endpoint for the frontend to poll for job status updates."""
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    try:
        video_key = f"jobs/{job_id}/output/final_video.mp4"
        s3_client.head_object(Bucket=AWS_S3_BUCKET_NAME, Key=video_key)
        video_url = f"https://{AWS_S3_BUCKET_NAME}.s3.amazonaws.com/{video_key}"
        return jsonify({"status": "done", "downloadUrl": video_url})
    except s3_client.exceptions.ClientError:
        pass # Video not found, check for status file
    try:
        status_key = f"jobs/{job_id}/status.json"
        status_obj = s3_client.get_object(Bucket=AWS_S3_BUCKET_NAME, Key=status_key)
        return jsonify(json.loads(status_obj['Body'].read().decode('utf-8')))
    except s3_client.exceptions.ClientError:
        return jsonify({"status": "pending", "message": "Job is queued and waiting for worker..."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

