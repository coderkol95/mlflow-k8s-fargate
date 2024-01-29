import re
import os
from dotenv import load_dotenv
load_dotenv()

new_image_version=os.environ["IMAGE_TAG"]

with open("../k8s-deployment.yaml","r") as f:
    k8s_yaml=f.read()

image_version=re.findall('\d+\.\d+\.\d+',k8s_yaml)[0]

k8s_yaml_updated=re.sub(image_version,new_image_version,k8s_yaml)

with open("../k8s-deployment.yaml","w") as f:
    f.write(k8s_yaml_updated)