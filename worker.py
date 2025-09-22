import os
import sys
import json
import subprocess
import time
import boto3
import requests
from urllib.parse import urlparse
import shutil
import textwrap
from bs4 import BeautifulSoup

# --- Configuration ---
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME")
AWS_S3_REGION = os.environ.get("AWS_S3_REGION", "us-east-1")
JOB_ID = os.environ.get("JOB_ID")

# --- Helper Functions ---
def wrap_text(text, width=40):
    return "\n".join(textwrap.wrap(text, width=width))

def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

# --- Scraping Helpers ---
def fetch_reddit_content(url):
    if not url.endswith(".json"):
        if url.endswith("/"):
            url = url[:-1]
        url += ".json"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    try:
        post_data = data[0]['data']['children'][0]['data']
        title = post_data.get('title', '')
        body = post_data.get('selftext', '')
        comments = [c['data']['body'] for c in data[1]['data']['children'] if c['kind'] == 't1']
        return {"title": title, "body": body, "comments": comments}
    except (KeyError, IndexError):
        return {"title": "", "body": "", "comments": []}

def fetch_static_site(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    title = soup.title.string if soup.title else ''
    paragraphs = "\n".join([p.get_text() for p in soup.find_all('p')])
    return {"title": title, "body": paragraphs}

def scrape_url(url):
    try:
        if "reddit.com" in url:
            return fetch_reddit_content(url)
        else:
            return fetch_static_site(url)
    except Exception as e:
        return {"title": "", "body": f"Failed to scrape: {e}", "comments": []}

# --- Core Video Processing ---
def process_job(job_data):
    # Example: Download assets from S3
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_S3_REGION
    )

    job_id = job_data.get("job_id", JOB_ID)
    if not job_id:
        print("No job ID provided.")
        sys.exit(1)

    job_prefix = f"jobs/{job_id}/"
    local_dir = f"/tmp/{job_id}"
    os.makedirs(local_dir, exist_ok=True)

    # Download all input assets
    objects = s3.list_objects_v2(Bucket=AWS_S3_BUCKET_NAME, Prefix=job_prefix)
    if 'Contents' not in objects:
        print("No assets found in S3 for this job.")
        sys.exit(1)

    for obj in objects['Contents']:
        key = obj['Key']
        if key.endswith('/'):
            continue
        local_path = os.path.join(local_dir, os.path.basename(key))
        s3.download_file(AWS_S3_BUCKET_NAME, key, local_path)

    # Load job.json
    job_file = os.path.join(local_dir, "job.json")
    if not os.path.exists(job_file):
        print("job.json missing!")
        sys.exit(1)

    with open(job_file, "r") as f:
        job_config = json.load(f)

    # Example scraping if job contains URLs
    if "source_url" in job_config:
        scraped = scrape_url(job_config["source_url"])
        print(f"Scraped Content: {scraped['title']} ({len(scraped['body'])} chars)")

    # --- FFmpeg Processing Logic ---
    output_file = os.path.join(local_dir, f"{job_id}.mp4")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", os.path.join(local_dir, "audio.mp3"),
        "-i", os.path.join(local_dir, "image.png"),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        output_file
    ]

    print(f"Running FFmpeg: {' '.join(ffmpeg_cmd)}")
    subprocess.run(ffmpeg_cmd, check=True)

    # Upload final video
    final_key = f"jobs/{job_id}/{job_id}.mp4"
    s3.upload_file(output_file, AWS_S3_BUCKET_NAME, final_key)
    print(f"Uploaded final video to s3://{AWS_S3_BUCKET_NAME}/{final_key}")

    # Cleanup (optional)
    shutil.rmtree(local_dir)

# --- Main Execution ---
if __name__ == "__main__":
    if not JOB_ID:
        print("JOB_ID not provided.")
        sys.exit(1)

    # Simulate loading job.json from S3 and processing
    job_data = {"job_id": JOB_ID}
    process_job(job_data)
