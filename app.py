
import sys
import os
from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
from network.gazenet import GazeNet
import random
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.use_deterministic_algorithms(True)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

app = Flask(__name__)

# تحميل الموديل
model = GazeNet(backbone='ResNet-34', view='single', pretrained=False)
#model_path = r'C:\Users\yassm\Downloads\multi-view-gaze-master\multi-view-gaze-master\multi-view-gaze-master\exps\2025-06-04-15-00\ckpts/model_best.pth.tar'
model_path = 'model_best.pth.tar'
state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

#state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
model.load_state_dict(state_dict, strict=False)
model.eval()

# حفظ القيم الأخيرة للتركيز
focus_history = []

# تقسيم الصورة لنصفين (عين يمنى ويسرى)
def split_image(image):
    width, height = image.size
    left_eye = image.crop((0, 0, width // 2, height))
    right_eye = image.crop((width // 2, 0, width, height))
    return left_eye, right_eye

# تجهيز صور العينين
def prepare_eye_tensors(left_eye_img, right_eye_img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    left_eye_tensor = transform(left_eye_img).unsqueeze(0)
    right_eye_tensor = transform(right_eye_img).unsqueeze(0)
    return left_eye_tensor, right_eye_tensor

# تحليل زاوية النظر وتحديد إذا كان الطفل مركز
def is_focused(yaw, pitch, threshold=1):
    return abs(yaw) < threshold and abs(pitch) < threshold


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file).convert('RGB')

    # قسم الصورة لنصفي العينين
    left_eye_img, right_eye_img = split_image(image)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    left_eye_tensor = transform(left_eye_img).unsqueeze(0)  # [1,3,224,224]
    right_eye_tensor = transform(right_eye_img).unsqueeze(0)  # [1,3,224,224]
    eye_location_tensor = torch.zeros(1, 24)  # dummy input

    inputs = (left_eye_tensor, right_eye_tensor, eye_location_tensor)

    with torch.no_grad():
        output = model(*inputs)  # output: [1, 2] → yaw, pitch
        yaw, pitch = output[0].tolist()
        print(f"Yaw: {yaw}, Pitch: {pitch}")

        # نحكم هل الطفل مركز ولا لا
        is_focused = (-5.3 <= yaw <= 5.3) and (-10 <= pitch <= 10)




    return jsonify({
        'yaw': yaw,
        'pitch': pitch,
        'focused': is_focused
    })


#if __name__ == '__main__':
#    app.run(host='0.0.0.0', port=5000, debug=True)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
