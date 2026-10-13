"""R2 (S3-compatible) helpers shared by the Kaggle jobs. Credentials come from a private Kaggle dataset."""
import glob, os, boto3, botocore

def env():
    for f in glob.glob('/kaggle/input/**/r2.env', recursive=True) + [os.path.expanduser('~/.secrets/r2.env')]:
        if os.path.exists(f):
            return dict(l.strip().split('=', 1) for l in open(f) if '=' in l)
    raise SystemExit('r2.env not found')

E = env()
BUCKET = E['R2_BUCKET']
s3 = boto3.client('s3', endpoint_url=E['R2_ENDPOINT'], aws_access_key_id=E['R2_ACCESS_KEY_ID'],
                  aws_secret_access_key=E['R2_SECRET_ACCESS_KEY'], region_name='auto',
                  config=botocore.config.Config(retries={'max_attempts': 8, 'mode': 'adaptive'}))

def put(key, data: bytes): s3.put_object(Bucket=BUCKET, Key=key, Body=data)
def put_file(key, path): s3.upload_file(path, BUCKET, key)
def get(key) -> bytes: return s3.get_object(Bucket=BUCKET, Key=key)['Body'].read()
def get_file(key, path): s3.download_file(BUCKET, key, path)
def exists(key):
    try: s3.head_object(Bucket=BUCKET, Key=key); return True
    except botocore.exceptions.ClientError: return False
def ls(prefix):
    out, tok = [], None
    while True:
        kw = dict(Bucket=BUCKET, Prefix=prefix, **({'ContinuationToken': tok} if tok else {}))
        r = s3.list_objects_v2(**kw); out += [o['Key'] for o in r.get('Contents', [])]
        if not r.get('IsTruncated'): return out
        tok = r['NextContinuationToken']
def delete(key): s3.delete_object(Bucket=BUCKET, Key=key)
