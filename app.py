from flask import Flask, jsonify, request
from NaiveBayes.naive_bayes import predict_statistic
from INER.rule_based import predict_rule_based
from flask_mysqldb import MySQL
import pandas as pd


app = Flask(__name__)

# app configuration
app.config["MYSQL_HOST"] = 'localhost'
app.config["MYSQL_USER"] = 'root'
app.config["MYSQL_PASSWORD"] = ''
app.config["MYSQL_DB"] = 'NERDISASTER'


# mysql configuration
mysql = MySQL(app=app)


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

@app.route("/rule/upload",methods=['POST'])
def upload_dataset():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'message': 'No data provided'}), 400
        df = pd.DataFrame(data=data)
        print(df)
        df.to_sql(name='TrainTable', con=engine,if_exists='replace', index=False)

        return jsonify({'message': 'DataFrame successfully stored in the database'}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
