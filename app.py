from flask import Flask, jsonify, request
from NaiveBayes.naive_bayes import predict_statistic
from flask_mysqldb import MySQL
app = Flask(__name__)
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_PORT'] = 8889
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'db_ner'
mysql = MySQL(app)
@app.route("/statistic",methods=['GET'])
def statistic():
    text = request.args.get('text')
    response = {'zresult':predict_statistic(text)}
    return jsonify(response)

# @app.route('/data/<int:id>',methods=['GET'])
# def get_data_by_id(id):
#     cur = mysql.connection.cursor()
#     cur.execute('''SELECT * FROM test_aja where id = %s''',(id,))
#     data = cur.fetchall()
#     cur.close()
#     return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
