import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import os

# --- Configuration ---
# Use environment variables for table name for better flexibility
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "VideoGeneratorCodes")
dynamodb_table = None

def init_aws_credentials(access_key, secret_key, region='us-east-1'):
    """Initializes the module with AWS credentials and the DynamoDB table resource."""
    global dynamodb_table
    
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    dynamodb = session.resource('dynamodb')
    dynamodb_table = dynamodb.Table(DYNAMODB_TABLE_NAME)

def register_new_code(email, code, tier="one_time"):
    """Adds a new code to DynamoDB based on the specified tier."""
    # This function would be called after a successful payment for paid tiers
    item = {
        "code": code,
        "email": email,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_used": None
    }

    if tier == "unlimited":
        item.update({"type": "unlimited"})
    elif tier == "monthly":
        item.update({
            "type": "monthly",
            "max_uses": 30, # Or 60, depending on the specific plan
            "used_count": 0,
            "usage_period_start": datetime.now(timezone.utc).isoformat()
        })
    else: # Default to the free "one_time" code
        item.update({
            "type": "one_time",
            "max_uses": 3,
            "used_count": 0,
        })
    
    try:
        dynamodb_table.put_item(
            Item=item,
            ConditionExpression='attribute_not_exists(code)'
        )
        print(f"Successfully registered '{tier}' code '{code}' in DynamoDB.")
        return {"success": True, "message": "Code registered successfully."}
    except ClientError as e:
        if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
            return {"success": False, "message": "This code has already been registered."}
        print(f"Error saving code to DynamoDB: {e}")
        return {"success": False, "message": "Failed to save code to database."}

def validate_code(code):
    """Validates if a code is valid, handling monthly resets if necessary."""
    try:
        response = dynamodb_table.get_item(Key={'code': code})
        code_info = response.get('Item')

        if not code_info:
            return {"valid": False, "message": "Invalid code."}

        code_type = code_info.get("type")
        
        if code_type == "unlimited":
            return {"valid": True, "message": "Unlimited access code is valid."}

        # Handle monthly reset logic
        if code_type == "monthly":
            now = datetime.now(timezone.utc)
            period_start_str = code_info.get("usage_period_start")
            period_start = datetime.fromisoformat(period_start_str)
            
            # Check if one month has passed
            if now >= period_start + relativedelta(months=1):
                # If reset is needed, the user has all their uses available
                remaining = int(code_info.get("max_uses", 0))
                return {"valid": True, "message": f"Code is valid. {remaining} uses remaining (new cycle)."}

        # For one_time codes and non-reset monthly codes
        used = int(code_info.get("used_count", 0))
        max_uses = int(code_info.get("max_uses", 0))
        if used < max_uses:
            remaining = max_uses - used
            return {"valid": True, "message": f"Code is valid. {remaining} uses remaining."}
        else:
            return {"valid": False, "message": "This code has no uses left for the current period."}

    except ClientError as e:
        print(f"Error validating code from DynamoDB: {e}")
        return {"valid": False, "message": "Error validating code."}

def update_code_usage(code):
    """Atomically increments usage, handling monthly resets before updating."""
    try:
        response = dynamodb_table.get_item(Key={'code': code})
        code_info = response.get('Item')

        if not code_info:
            return {"success": False, "message": "Code not found."}

        code_type = code_info.get("type")
        if code_type == "unlimited":
            return {"success": True, "message": "Code usage updated for unlimited code."}

        now_iso = datetime.now(timezone.utc).isoformat()
        
        # Determine if a monthly reset is needed
        reset_needed = False
        if code_type == "monthly":
            now = datetime.now(timezone.utc)
            period_start = datetime.fromisoformat(code_info.get("usage_period_start"))
            if now >= period_start + relativedelta(months=1):
                reset_needed = True

        if reset_needed:
            # Atomically reset the count to 1 and update the period start time
            dynamodb_table.update_item(
                Key={'code': code},
                UpdateExpression="SET used_count = :one, last_used = :ts, usage_period_start = :ts",
                ExpressionAttributeValues={
                    ':one': 1,
                    ':ts': now_iso,
                }
            )
            print(f"Successfully reset and updated usage for monthly code '{code}'.")
            return {"success": True, "message": "Code usage updated."}
        else:
            # Atomically increment the count if it's less than the max
            dynamodb_table.update_item(
                Key={'code': code},
                UpdateExpression="SET used_count = used_count + :inc, last_used = :ts",
                ConditionExpression="used_count < max_uses",
                ExpressionAttributeValues={
                    ':inc': 1,
                    ':ts': now_iso,
                }
            )
            print(f"Successfully incremented usage for code '{code}'.")
            return {"success": True, "message": "Code usage updated."}

    except ClientError as e:
        if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
             return {"success": False, "message": "Code has no uses left or condition failed."}
        print(f"Error updating code usage in DynamoDB: {e}")
        return {"success": False, "message": "Failed to update code usage."}