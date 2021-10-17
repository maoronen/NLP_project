import numpy as np
from scipy.special import softmax
from tensorflow.keras import optimizers, losses
from transformers import BertTokenizer, TFBertForSequenceClassification
from flask import Flask, request, render_template


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained("bert_model", local_files_only=True)
opt = optimizers.Adam(learning_rate=3e-5)
scce = losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=opt, loss=scce, metrics='accuracy')
input_names = ['input_ids', 'token_type_ids', 'attention_mask']

app = Flask(__name__, template_folder='template')


@app.route('/')
def my_index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict_tweet():
    """
    This function takes arguments from the URL bar, creates an array to predict on,
    and lastly uses the pickled model to make a prediction which is returned to the client
    :return: string of prediction ('0' or '1')
    """
    input_text = request.form['input_text']
    tokenized_tweet = tokenizer(input_text)
    logits = model.predict({k: np.array(tokenized_tweet[k])[None] for k in input_names})[0]
    scores = softmax(logits)
    pred = round(100*scores.flatten()[1], 2)
    # return render_template('index.html', prediction=pred)
    if pred >= 50:
        return render_template('index.html', prediction=f'Emergency; confidence ({pred}%)')
    else:
        return render_template('index.html', prediction=f'Non-emergency; confidence ({100-pred}%)')


if __name__ == '__main__':
    app.run(debug=True)
