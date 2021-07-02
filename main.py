from __future__ import print_function
from flask import Flask,request,Response,jsonify
import projects.UniDet.demo.demo as demo
import json

app = Flask(__name__)


# constants
WINDOW_NAME = "COCO detections"



@app.route('/')
def home():
    return 'Hello world'

@app.route('/test')
def test():
    # args = f'python projects/UniDet/demo/demo.py --config-file projects/UniDet/configs/Unified_learned_OCIM_R50_6x+2x.yaml --input {file} --opts MODEL.WEIGHTS models/Unified_learned_OCIM_R50_6x+2x.pth'
    image = request.args.get('file',None)
    if image == None:
        return "There is no Image"
    jsonTemp = demo.predict(image)
    return json.dumps(jsonTemp)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000,debug=True)

#python projects/UniDet/demo/demo.py --config-file projects/UniDet/configs/Unified_learned_OCIM_R50_6x+2x.yaml --input images/*.jpg --opts MODEL.WEIGHTS models/Unified_learned_OCIM_R50_6x+2x.pth