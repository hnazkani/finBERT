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
    
    '''
    POST request:
        url: http://0.0.0.0:8080/
        body: {"text": "Later that day Apple said it was revising down its earnings expectations in the fourth quarter of 2018, largely because of lower sales and signs of economic weakness in China. The news rapidly infected financial markets. Apple’s share price fell by around 7% in after-hours trading and the decline was extended to more than 10% when the market opened. The dollar fell by 3.7% against the yen in a matter of minutes after the announcement, before rapidly recovering some ground. Asian stockmarkets closed down on January 3rd and European ones opened lower. Yields on government bonds fell as investors fled to the traditional haven in a market storm."}
    '''