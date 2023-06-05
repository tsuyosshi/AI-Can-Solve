from flask import *
import cv2
import numpy as np
import keras
import pandas as pd
import json
import base64
import pickle
import math
from read_pazzle import read_image, predict_nums
from solver import DFS, BFS, BeamSearch

app = Flask(__name__)
model = None

def load_model():
    global model
    print(" * Loading pre-trained model ...")
    model = keras.models.load_model("./work/model")
    print(' * Loading end')

@app.route("/")
def hello_world():
    return "Hello, World!"

@app.route("/read_pazzle", methods=["GET", "POST"])
def read_pazzle():
    data = json.loads(request.data.decode("utf-8"))



    img_bin = data['image']
    decoded = base64.b64decode(img_bin)
    jpg = np.asarray(bytearray(decoded), dtype=np.uint8)
    img = cv2.imdecode(jpg, 0)
    np.save('./work/img.npy', img)
    sz, df = predict_nums(img)
    nums = df['value'].to_list()
    solver = BeamSearch(numbers=nums, W=math.sqrt(sz), H=math.sqrt(sz))
    ans = solver.run()
    payload = {
        'size': sz,
        'numbers': nums,
        'ans': ans
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    load_model()
    print(" * Flask starting server...")
    app.run(host='0.0.0.0', port=80)