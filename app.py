from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import os

# 모델 로드
model_path = 'pill_resnet152_dataclass01_aug0.pt'
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

app = Flask(__name__)

# 이미지 전처리
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_bytes)
    return transform(image).unsqueeze(0)

# 예측 함수
def get_prediction(image_tensor):
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        img_tensor = transform_image(file)
        prediction = get_prediction(img_tensor)
        return jsonify({'pill_class': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
