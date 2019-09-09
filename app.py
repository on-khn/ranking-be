import time
from flask import Flask, request
from flask_cors import CORS
from services import training

app = Flask(__name__)
cors = CORS(app)

@app.route("/train", methods=['GET'])
def train():
    # measure the time
    start = time.time()
    ranker, train_spec, eval_spec = training._trainer("./resources/train.tfrecords", "./resources/test.tfrecords")
    if eval_spec == None:
        return {'message': 'training had an error'}, 500
    spent_time = time.time()-start
    return {'status': 'success', 'data': spent_time}, 200