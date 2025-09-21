import os
import sys
import json
import subprocess
import time
import boto3
from urllib.parse import urlparse
import shutil
import textwrap
import base64
import requests
import re
from bs4 import BeautifulSoup

# --- Part 1: Configuration ---
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME")
JOB_ID = os.environ.get("JOB_ID")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# ----------------------------
# Utility Functions
# ----------------------------

def wrap_text(text, width=40):
    if not text:
        return ""
    words = text.split()
    lines = []
    current = []
    current_len = 0
    for w in words:
        if current_len + len(w) + (1 if current else 0) > width:
            lines.append(" ".join(current))
            current = [w]
            current_len = len(w)
        else:
            current.append(w)
            current_len += len(w) + (1 if current_len else 0)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)

def format_time(seconds):
    try:
        seconds = int(seconds)
    except Exception:
        seconds = 0
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

# ----------------------------
# Reddit Scraping (Updated to JSON API)
# ----------------------------

def process_reddit_thread(url, api_key=None):
    print(f"Processing Reddit URL via JSON API: {url}")
    headers = {
        'User-Agent': os.environ.get(
            'SCRAPER_USER_AGENT',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/116.0.0.0 Safari/537.36'
        )
    }
    session = requests.Session()
    session.headers.update(headers)

    parsed = urlparse(url)
    if not parsed.scheme:
        url = 'https://' + url
        parsed = urlparse(url)
    if not parsed.path.endswith('.json'):
        json_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}.json"
        if parsed.query:
            json_url += '?' + parsed.query
    else:
        json_url = url

    max_retries = 4
    backoff = 1
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = session.get(json_url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) >= 1:
                post_part = data[0].get('data', {}).get('children', [])
                if post_part:
                    post = post_part[0].get('data', {})
                    title = post.get('title', '')
                    selftext = post.get('selftext', '')
                    author = post.get('author', '')
                else:
                    title = ''
                    selftext = ''
                    author = ''
                comments = []
                if len(data) > 1:
                    comments_part = data[1].get('data', {}).get('children', [])
                    for c in comments_part:
                        if c.get('kind') != 't1':
                            continue
                        cdata = c.get('data', {})
                        body = cdata.get('body', '')
                        if body:
                            comments.append(body)
                        if len(comments) >= 5:
                            break
                full_text = f"Title: {title}. Author: {author}. Post: {selftext}"
                if comments:
                    full_text += "\n\nTop comments:\n" + "\n---\n".join(comments)
                print(f"Extracted Reddit content length: {len(full_text)}")
                return full_text
            else:
                raise Exception("Unexpected JSON structure from Reddit.")
        except Exception as e:
            last_exc = e
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            time.sleep(backoff)
            backoff *= 2
    raise Exception(f"Failed to fetch Reddit content after {max_retries} attempts: {last_exc}")

# ----------------------------
# Generic URL Scraping (with Multiple Fallbacks)
# ----------------------------

def process_generic_url(url, api_key=None):
    print(f"Processing generic URL: {url}")
    headers = {
        'User-Agent': os.environ.get(
            'SCRAPER_USER_AGENT',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/116.0.0.0 Safari/537.36'
        )
    }
    session = requests.Session()
    session.headers.update(headers)

    def allowed_by_robots(u):
        try:
            parsed = urlparse(u)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            r = session.get(robots_url, timeout=6)
            if r.status_code == 200:
                txt = r.text
                if re.search(r"(?m)Disallow:\s*/\s*$", txt, re.I):
                    return False
            return True
        except Exception:
            return True

    if not urlparse(url).scheme:
        url = "https://" + url

    if not allowed_by_robots(url):
        raise Exception("Scraping disallowed by robots.txt")

    def try_requests_plain(u):
        max_retries = 3
        backoff = 1
        last_exc = None
        for attempt in range(max_retries):
            try:
                resp = session.get(u, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.content, 'html.parser')
                for el in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
                    el.decompose()
                title = soup.title.string.strip() if soup.title and soup.title.string else u
                page_text = soup.get_text(separator=' ', strip=True)[:10000]
                if page_text and len(page_text.strip()) > 50:
                    return title, page_text
                meta = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
                if meta and meta.get('content'):
                    return title, meta.get('content').strip()[:10000]
                last_exc = Exception("No substantial text found")
            except Exception as e:
                last_exc = e
                print(f"Plain request attempt {attempt+1} failed: {e}")
                time.sleep(backoff)
                backoff *= 2
        raise last_exc

    def try_jina(u):
        try:
            normalized = u
            if normalized.startswith("https://"):
                normalized = normalized[len("https://"):]
            if normalized.startswith("http://"):
                normalized = normalized[len("http://"):]
            jina_url = "https://r.jina.ai/http://" + normalized
            r = session.get(jina_url, timeout=15)
            r.raise_for_status()
            text = r.text
            if text and len(text.strip()) > 50:
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                title = lines[0] if lines else u
                return title, "\n".join(lines[:500])[:10000]
        except Exception as e:
            print(f"Jina.ai extraction failed: {e}")
        return None, None

    def try_external_api(u):
        api = os.environ.get('SCRAPING_API_URL')
        if not api:
            return None, None
        try:
            r = session.get(api, params={'url': u}, timeout=30)
            r.raise_for_status()
            try:
                j = r.json()
                text = j.get('text') or j.get('content') or ''
                if text and len(text.strip()) > 50:
                    return u, text[:10000]
            except Exception:
                soup = BeautifulSoup(r.content, 'html.parser')
                for el in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    el.decompose()
                title = soup.title.string if soup.title else u
                page_text = soup.get_text(separator=' ', strip=True)[:10000]
                if page_text and len(page_text.strip()) > 50:
                    return title, page_text
        except Exception as e:
            print(f"External scraping API failed: {e}")
        return None, None

    try:
        title, text = try_requests_plain(url)
    except Exception as e:
        print(f"Plain requests failed: {e}")
        title, text = None, None

    if not text:
        title, text = try_jina(url)

    if not text:
        title, text = try_external_api(url)

    if not text:
        raise Exception("Failed to extract text from URL.")

    print(f"Extracted title: {title}, text length: {len(text)}")
    return title + "\n\n" + text

# ----------------------------
# Video Processing (Original)
# ----------------------------

def process_job(job_data):
    print(f"Processing job {job_data['job_id']}")
    input_video = job_data.get('input_video')
    output_video = "/tmp/output.mp4"

    # Example of unchanged ffmpeg logic:
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", "scale=1280:720",
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        output_video
    ]
    print(f"Running ffmpeg: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return output_video

# ----------------------------
# Main Execution (Original)
# ----------------------------

if __name__ == "__main__":
    print(f"Worker started for job {JOB_ID}")

    # Download job data from S3
    job_key = f"jobs/{JOB_ID}/job.json"
    local_job_path = "/tmp/job.json"
    s3.download_file(AWS_S3_BUCKET_NAME, job_key, local_job_path)

    with open(local_job_path, "r") as f:
        job_data = json.load(f)

    try:
        result_path = process_job(job_data)

        # Upload result
        result_key = f"jobs/{JOB_ID}/result.mp4"
        s3.upload_file(result_path, AWS_S3_BUCKET_NAME, result_key)
        print("Upload complete")
    except Exception as e:
        print(f"Job failed: {e}")
        sys.exit(1)
