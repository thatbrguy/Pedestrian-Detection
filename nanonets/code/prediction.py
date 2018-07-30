import requests
import os
import sys
import cv2
import time
import pickle

model_id = os.environ.get('NANONETS_MODEL_ID')
api_key = os.environ.get('NANONETS_API_KEY')

if api_key is None:
    print('Run the following command on your terminal first:')
    print('export NANONETS_API_KEY="INPUT_YOUR_API_KEY_HERE"')
    exit(1)

if model_id is None:
    print('Run the following command on your terminal first:')
    print('export NANONETS_MODEL_ID="INPUT_YOUR_MODEL_ID_HERE"')
    exit(1)

if not os.path.exists(os.path.join(os.getcwd(), 'test_images')):
    print('Create a directory named "test_images" and put all your images there')
    exit(1)

if not os.path.exists(os.path.join(os.getcwd(), 'nanonets_output')):
    os.mkdir('nanonets_output')

url = 'https://app.nanonets.com/api/v2/ObjectDetection/Model/' + model_id + '/LabelFile/'

font = cv2.FONT_HERSHEY_SIMPLEX
color = (125,254,3)
count_dict = {}
cwd = os.getcwd()
files = sorted([os.path.join(cwd, 'test_images', file) for file in os.listdir('test_images')])

for i, image_path in enumerate(files):

    begin = time.time()
    data = {'file': open(image_path, 'rb'),    'modelId': ('', model_id)}
    response = requests.post(url, auth=requests.auth.HTTPBasicAuth(api_key, ''), files=data)

    if response.status_code == 200:
        image = cv2.imread(image_path, 1)
        output = response.json()
        boxes = output['result'][0]['prediction']
        count = len(boxes)
        for box in boxes:
            xmax, xmin, ymax, ymin, score = box['xmax'], box['xmin'], box['ymax'], box['ymin'], box['score']
            cv2.rectangle(image, (xmin, ymin-10), (xmin+75, ymin), color, -1)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 3)
            cv2.putText(image, 'pedestrian: %d%%'%(score * 100), (xmin, ymin), font, fontScale=0.3, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)

        cv2.imwrite(os.path.join('nanonets_output', os.path.basename(image_path)), image)
        count_dict[i] = count
        print('File:', os.path.basename(image_path), 'Pedestrian Count:', count)

    else:
        print('Error:', response.status_code, 'File:', os.path.basename(image_path))

with open('nanonets_count.pickle', 'wb') as handle:
    pickle.dump(count_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
