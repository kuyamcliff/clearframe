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

def retry(fn, *a, **kw):
    """R2 throttles bursts ("Reduce your concurrent request rate"); back off and retry for up to ~5 minutes."""
    import time, random
    for i in range(10):
        try: return fn(*a, **kw)
        except botocore.exceptions.ClientError as e:
            code = e.response.get('Error', {}).get('Code', '')
            if code not in ('ServiceUnavailable', 'SlowDown', 'InternalError', '503', '500') or i == 9: raise
        except (botocore.exceptions.EndpointConnectionError, botocore.exceptions.ConnectionClosedError, botocore.exceptions.ReadTimeoutError):
            if i == 9: raise
        time.sleep(min(60, 2 ** i) * (0.5 + random.random()))

def list_page(prefix, token=None):
    return retry(s3.list_objects_v2, Bucket=BUCKET, Prefix=prefix, **({'ContinuationToken': token} if token else {}))

def put(key, data: bytes): retry(s3.put_object, Bucket=BUCKET, Key=key, Body=data)
def put_file(key, path): retry(s3.upload_file, path, BUCKET, key)
def get(key) -> bytes: return retry(lambda: s3.get_object(Bucket=BUCKET, Key=key)['Body'].read())
def get_file(key, path): retry(s3.download_file, BUCKET, key, path)
def exists(key):
    try: retry(s3.head_object, Bucket=BUCKET, Key=key); return True
    except botocore.exceptions.ClientError: return False
def ls(prefix):
    out, tok = [], None
    while True:
        r = list_page(prefix, tok); out += [o['Key'] for o in r.get('Contents', [])]
        if not r.get('IsTruncated'): return out
        tok = r['NextContinuationToken']
def delete(key): retry(s3.delete_object, Bucket=BUCKET, Key=key)
