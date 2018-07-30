import requests, os, json

url = "https://app.nanonets.com/api/v2/ObjectDetection/Model/"
api_key = os.environ.get('NANONETS_API_KEY')

payload = "{\"categories\" : [\"pedestrian\"]}"
headers = {'Content-Type': "application/json",}

response = requests.request("POST", url, headers=headers, auth=requests.auth.HTTPBasicAuth(api_key, ''), data=payload)

model_id = json.loads(response.text)["model_id"]
os.environ['NANONETS_MODEL_ID'] = model_id

print("NEXT RUN: python ./code/upload-training.py")