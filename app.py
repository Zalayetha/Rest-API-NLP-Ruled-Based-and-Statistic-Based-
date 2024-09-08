from flask import Flask, jsonify, request
from NaiveBayes.naive_bayes import predict_statistic
from INER.rule_based import predict_rule_based
from db.conn import getdb,close_db
import pandas as pd
from stopwords.stopwords import getStopwords
from normalization.normalization import getNormalization
from classStatistic.classStatistic import getClassStatistic
from preprocessingStatistic.preprocessingStatistic import insertPreprocessingStatistic,getPreprocessingStatistic
import json

app = Flask(__name__)
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_PORT'] = 8888
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'santana-db'


@app.route("/statistic",methods=['GET'])
def statistic():
    text = request.args.get('text')
    response = {'zresult':predict_statistic(teks=text)}
    return jsonify(response)

@app.route("/rule",methods=['GET'])
def rule():
    text = request.args.get('text')
    response = {'zresult':predict_rule_based(text=text)}
    return jsonify(response)

@app.route("/stopwords",methods=['GET'])
def stopwords():
    try:
        connection = getdb()
        stopwords = getStopwords(connection)
        return jsonify({"responseStatus": True,'responseMessage': 'Successfully retrieved data!',"responseBody":stopwords}), 200
    except Exception as e:
        return jsonify({'message': str(e)},500)

@app.route("/normalization",methods=['GET'])
def normalization():
    try:
        connection = getdb()
        normalization = getNormalization(connection)
        return jsonify({"responseStatus": True,'responseMessage': 'Successfully retrieved data!',"responseBody":normalization}), 200
    except Exception as e:
        return jsonify({'message': str(e)},500)

@app.route("/class-statistic",methods=['GET'])
def classStatistic():
    try:
        connection = getdb()
        classStatistic = getClassStatistic(connection)
        return jsonify({"responseStatus": True,'responseMessage': 'Successfully retrieved data!',"responseBody":classStatistic}), 200
    except Exception as e:
        return jsonify({'message': str(e)},500)

@app.route("/statistic/bio-labeling",methods=['POST','GET'])
def saveBioLabelingStatistic():
    try:
        connection = getdb()
        if request.method == 'POST':
            data = request.json
            # convert json to dataframe
            df = pd.read_json(data)
            insertPreprocessingStatistic(dataframe=df,connection=connection)
            return jsonify({"responseStatus": True,'responseMessage': 'Successfully insert data!'}), 200
        elif request.method == 'GET':
            data = getPreprocessingStatistic(connection=connection)
            return  jsonify({"responseStatus": True,'responseMessage': 'Successfully retrieved data!',"responseBody":data}), 200
    except Exception as e:
        return jsonify({"responseStatus": False,'responseMesage': str(e)},500)
    

if __name__ == '__main__':
    app.run(debug=True)
