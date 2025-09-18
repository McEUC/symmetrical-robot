module.exports = {
  apps : [{
    name   : "backrooms-app",
    script : "/home/evans_malcolmc/backrooms_generator/venv/bin/gunicorn",
    args   : "--workers 1 --bind 0.0.0.0:5000 --timeout 300 app:app",
    cwd    : "/home/evans_malcolmc/backrooms_generator/",
    env    : {
      "AWS_ACCESS_KEY_ID": "AKIAQVXHQ2OAQ3UAI3FU",
      "AWS_SECRET_ACCESS_KEY": "ssr+n4PDaix9sUomw8Iv+wi8teslAo54EmGWjD40",
      "AWS_S3_BUCKET_NAME": "backrooms-app",
      "AWS_S3_REGION": "us-east-1" 
    }
  }]
}
