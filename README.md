# CV
---
## Train
### Clone
```python
git clone https://github.com/khanhduy2311/FisheyeDetection
cd FisheyeDetection
```
### Build Docker Image
```python
docker build -t cv .
```

### Create Docker Container
```python
docker run -dit --name cv  -v ./:/workspace  --gpus all --ipc=host cv tail -f /dev/null
```
### Data Preparation
```python
chmod +x download_data.sh download_model.sh
./download_data.sh
./download_model.sh
```
### Yolo - Training
```python
chmod +x train_yolo.sh
./train_yolo.sh
```
### Dfine - Training
```python
chmod +x train_dfine.sh
./train_dfine.sh
```
## Inference
To perform inference using the trained model, follow the steps below:

1. **Navigate to the inference directory:**
   ```bash
   cd inference/
2. **Follow the instructions in the README.txt file inside the inference/ folder for detailed guidance**