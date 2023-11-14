from flask import Flask, render_template, request
import pickle
import numpy as np
Model = pickle.load(open('iris.pkl', 'rd'))

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def index():
    data1 = request.form['v1']
    data2 = request.form['v2']
    data3 = request.form['v3']
    data4 = request.form['v4']
    data5 = request.form['v5']
    data6 = request.form['v6']
    data7 = request.form['v7']
    data8 = request.form['v8']
    data9 = request.form['class']
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9]])
    pred = Model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)