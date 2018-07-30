# Pedestrian-Detector

-- Work in progress --

## Steps
### 1. For TensorFlow Object Detection API

### 2. For Nanonets
**Step 1: Clone the repo**
```
git clone https://github.com/NanoNets/object-detection-sample-python.git
cd object-detection-sample-python
sudo pip install requests
```
**Step 2: Get your free API Key**

Get your free API Key from http://app.nanonets.com/user/api_key

**Step 3: Set the API key as an Environment Variable**
```
export NANONETS_API_KEY=YOUR_API_KEY_GOES_HERE
```
**Step 4: Create a New Model**
```
python ./code/create-model.py
```
>**Note:** An environment variable NANONETS_MODEL_ID will be created in the previous step, with your model ID.

**Step 5: Upload the Training Data**

Place the training data in a folder named `images` and annotations in `annotations/json`
```
python ./code/upload-training.py
```
**Step 6: Train the Model**
```
python ./code/train-model.py
```
**Step 7: Get Model State**

The model takes ~2 hours to train. You will get an email once the model is trained. In the meanwhile you check the state of the model
```
python ./code/model-state.py
```
**Step 8: Make Predictions**

Create a folder named 'test_images' inside the 'nanonets' folder. Place the input images in this folder, and then run this command.
```
python ./code/prediction.py
```

## Results

## Notes
