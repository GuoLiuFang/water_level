import requests
import json
import numpy as np

url='http://127.0.0.1:5503'
files={'file': open('./test_image/00213_130.jpg','rb')}

r = requests.post(url, files=files)
data = json.loads(r.text)
# print(data)
mask = np.array(data["data"])
print(mask.shape)
