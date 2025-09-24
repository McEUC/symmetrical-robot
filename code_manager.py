import boto3
import json
from datetime import datetime
from botocore.exceptions import ClientError
import os

# --- CONFIGURATION ---
S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME", "default-bucket")
CODES_FILE_KEY = "codes.json"
ADMIN_EMAIL = "evans.malcolmc@gmail.com" # Your special admin email

s3_client = boto3.client('s3', 
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"), 
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
)

def get_codes_from_s3():
    """Fetches the codes.json file from S3."""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=CODES_FILE_KEY)
        return json.loads(response['Body'].read().decode('utf-8'))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return {"codes": {}, "emails": {}}
        raise

def save_codes_to_s3(data):
    """Saves the provided data object to codes.json in S3."""
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=CODES_FILE_KEY,
        Body=json.dumps(data, indent=2),
        ContentType='application/json'
    )

def register_new_code(email, code):
    """Handles logic for registering a new FREE code."""
    data = get_codes_from_s3()
    
    if email in data.get("emails", {}):
        return {"success": False, "message": "This email has already been registered.", "status_code": 409}

    is_admin = (email == ADMIN_EMAIL)
    
    new_code_entry = {
        "type": "unlimited" if is_admin else "one_time",
        "max_uses": float('inf') if is_admin else 3,
        "used_count": 0,
        "created_at": datetime.utcnow().isoformat(),
        "email": email,
        "last_used": None
    }
    
    data["codes"][code] = new_code_entry
    data.setdefault("emails", {})[email] = code
    
    save_codes_to_s3(data)
    return {"success": True, "message": "Code registered successfully.", "status_code": 201}

def create_paid_code(email, plan_type, paypal_subscription_id):
    """Handles logic for creating a new PAID monthly code."""
    data = get_codes_from_s3()

    if email in data.get("emails", {}):
        return {"success": False, "message": "This email is already associated with a code."}

    max_uses = 30 if plan_type == 'creator' else 60
    prefix = "MONTH30" if plan_type == 'creator' else "MONTH60"
    unique_id = (datetime.now().strftime('%Y%m%d%H%M%S') + os.urandom(4).hex()).upper()
    new_code = f"{prefix}-{unique_id}"

    new_code_entry = {
        "type": "monthly",
        "max_uses": max_uses,
        "used_count": 0,
        "created_at": datetime.utcnow().isoformat(),
        "email": email,
        "last_used": None,
        "paypal_subscription_id": paypal_subscription_id,
        "current_month": datetime.utcnow().strftime('%Y-%m')
    }

    data["codes"][new_code] = new_code_entry
    data.setdefault("emails", {})[email] = new_code # <-- Bug Fix: Was using 'code' which is not defined here.
    save_codes_to_s3(data)
    
    return {"success": True, "code": new_code}


def validate_code(code):
    """Validates a code and checks its usage."""
    data = get_codes_from_s3()
    code_info = data.get("codes", {}).get(code)

    if not code_info:
        return {"valid": False, "message": "Invalid code."}

    if code_info["type"] == "unlimited":
        return {"valid": True, "message": "Admin code is valid."}

    if code_info["type"] == "one_time":
        if code_info["used_count"] >= code_info["max_uses"]:
            return {"valid": False, "message": "This code has exceeded its usage limit."}
        return {"valid": True, "message": f"Code valid. {code_info['max_uses'] - code_info['used_count']} uses remaining."}

    if code_info["type"] == "monthly":
        current_month = datetime.utcnow().strftime('%Y-%m')
        if code_info.get("current_month") != current_month:
            # Reset monthly count if it's a new month
            code_info["used_count"] = 0
            code_info["current_month"] = current_month
            save_codes_to_s3(data)

        if code_info["used_count"] >= code_info["max_uses"]:
            return {"valid": False, "message": "Monthly usage limit reached."}
        
        remaining = code_info['max_uses'] - code_info['used_count']
        return {"valid": True, "message": f"Subscription valid. {remaining} uses remaining this month."}

    return {"valid": False, "message": "Unknown code type."}

def update_code_usage(code):
    """Increments the usage count for a given code."""
    data = get_codes_from_s3()
    code_info = data.get("codes", {}).get(code)

    if not code_info or code_info["type"] == "unlimited":
        return {"success": True} 

    code_info["used_count"] += 1
    code_info["last_used"] = datetime.utcnow().isoformat()
    save_codes_to_s3(data)
    
    return {"success": True, "message": "Usage updated."}

