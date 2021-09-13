from flask import Flask, redirect, url_for, request, render_template, jsonify, Response
import re
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        image_data = re.sub('^data:image/.+;base64,', '', request.json)
        pil_image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
        label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        label_names.sort()
        net = cv2.dnn.readNetFromONNX('cifar_classifier.onnx')
        img = cv2.resize(np.array(pil_image),(32,32))
        img = np.array([img]).astype('float64') / 255.0
        net.setInput(img)
        out = net.forward()
        index = np.argmax(out[0])
        label =  label_names[index].capitalize()
        return jsonify(result=label)
    return None

if __name__ == '__main__':
    app.run(debug=True)
