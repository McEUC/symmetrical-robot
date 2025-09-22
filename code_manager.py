import boto3
import json
from datetime import datetime
from botocore.exceptions import ClientError
import os

# --- Configuration ---
# These will be read from environment variables in app.py
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME")
CODES_FILE_KEY = "codes.json"

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def get_codes_data():
    """
    Retrieves the codes.json file from S3.
    If it doesn't exist, creates an empty structure.
    """
    try:
        response = s3_client.get_object(Bucket=AWS_S3_BUCKET_NAME, Key=CODES_FILE_KEY)
        return json.loads(response['Body'].read().decode('utf-8'))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print("codes.json not found in S3. Creating a new one.")
            initial_data = {"codes": {}}
            save_codes_data(initial_data)
            return initial_data
        else:
            raise

def save_codes_data(data):
    """Saves the provided data structure back to codes.json in S3."""
    s3_client.put_object(
        Bucket=AWS_S3_BUCKET_NAME,
        Key=CODES_FILE_KEY,
        Body=json.dumps(data, indent=2),
        ContentType='application/json'
    )
    print("Successfully saved codes.json to S3.")

def validate_code(code):
    """
    Validates a preview code, checking its type, usage, and monthly reset logic.
    Returns a dictionary with 'valid': boolean and a 'message'.
    """
    codes_data = get_codes_data()
    code_info = codes_data.get("codes", {}).get(code)

    if not code_info:
        return {"valid": False, "message": "Invalid code."}

    now = datetime.utcnow()
    
    if code_info["type"] == "one_time":
        if code_info["used_count"] >= code_info["max_uses"]:
            return {"valid": False, "message": "This code has already been used the maximum number of times."}
    
    elif code_info["type"] == "monthly":
        current_month_str = now.strftime("%Y-%m")
        # If the code's recorded month is not the current month, reset the count.
        if code_info.get("current_month") != current_month_str:
            code_info["used_count"] = 0
            code_info["current_month"] = current_month_str
            # We need to save the reset state immediately. This is a read-modify-write operation.
            # In a high-concurrency environment, this should be handled by a transactional database or locks.
            # For this system's scale, this is acceptable.
            save_codes_data(codes_data)

        if code_info["used_count"] >= code_info["max_uses"]:
            return {"valid": False, "message": "You have reached your monthly limit for this code."}
            
    else:
        return {"valid": False, "message": "Unknown code type."}

    return {"valid": True, "message": "Code is valid."}

def update_code_usage(code):
    """
    Increments the usage count for a given code. This is called after a successful video generation.
    """
    if not code:
        return {"success": True, "message": "No code provided; proceeding."}

    codes_data = get_codes_data()
    code_info = codes_data.get("codes", {}).get(code)

    if not code_info:
        # This is a safeguard; validation should have caught this.
        return {"success": False, "message": "Code not found for usage update."}

    # Increment usage and update timestamp
    code_info["used_count"] += 1
    code_info["last_used"] = datetime.utcnow().isoformat()

    # For monthly codes, ensure the month is set
    if code_info["type"] == "monthly" and not code_info.get("current_month"):
        code_info["current_month"] = datetime.utcnow().strftime("%Y-%m")

    save_codes_data(codes_data)
    return {"success": True, "message": f"Code {code} usage updated."}
