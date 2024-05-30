from flask import Flask, jsonify, request
from NaiveBayes.naive_bayes import predict_statistic
app = Flask(__name__)

@app.route("/statistic",methods=['GET'])
def statistic():
    text = request.args.get('text')
    response = {'zresult':predict_statistic(text)}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
