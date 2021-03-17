from flask import Flask
from flask_cors import CORS
import sys
import optparse
import time
from flask import request
import sys
from finbert.finbert import predict
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from transformers import AutoModelForSequenceClassification
import nltk

nltk.download('punkt')
app = Flask(__name__)
CORS(app)
start = int(round(time.time()))

model = AutoModelForSequenceClassification.from_pretrained('models/classifier_model/finbert-sentiment', num_labels=3, cache_dir=None)

@app.route("/",methods=['POST'])
def score():
    text=request.get_json()['text']
    result_df = predict(text, model)
    print (result_df)
    return(result_df.to_json(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
