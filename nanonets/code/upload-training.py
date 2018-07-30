import os, requests

pathToAnnotations = './annotations/json'
pathToImages = './images'
model_id = os.environ.get('NANONETS_MODEL_ID')
api_key = os.environ.get('NANONETS_API_KEY')
files = os.listdir(pathToAnnotations)
root = './annotations/json'

for i, name in enumerate(files, 1):
    annotation = open(os.path.join(root, name), "r")
    filePath = os.path.join(root, name)
    imageName, ext = name.split(".")
    imagePath = os.path.join(pathToImages, imageName + '.jpg')
    jsonData = annotation.read()
    url = 'https://app.nanonets.com/api/v2/ObjectDetection/Model/' + model_id + '/UploadFile/'
    data = {'file' :open(imagePath, 'rb'),  'data' :('', '[{"filename":"' + imageName+".jpg" + '", "object": '+ jsonData+'}]'),   'modelId' :('', model_id)}       
    response = requests.post(url, auth=requests.auth.HTTPBasicAuth(api_key, ''), files=data)
    print('Image:', os.path.basename(imagePath), 'Status:', response.status_code, 'Count:', i)

print("\n\n\nNEXT RUN: python ./code/train-model.py")