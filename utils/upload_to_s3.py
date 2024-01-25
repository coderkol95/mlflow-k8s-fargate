import boto3
import os
import sys

def upload_recursively_to_s3(destination,AK,SK):

    local_directory=destination
    client = boto3.client('s3',aws_access_key_id=AK, aws_secret_access_key=SK,region_name='us-east-1')

    bucket='mlops-optuna'

    for root, dirs, files in os.walk(local_directory):

        for filename in files:
            
            print("\n\n")
            # construct the full local path
            local_path = os.path.join(root, filename)
            # construct the full Dropbox path
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(destination, relative_path)
            try:
                print('Searching "%s" in "%s"' % (s3_path, bucket))
                client.head_object(Bucket=bucket, Key=s3_path)
            except:
                print("Uploading %s..." % s3_path)
                client.upload_file(local_path, bucket, s3_path)