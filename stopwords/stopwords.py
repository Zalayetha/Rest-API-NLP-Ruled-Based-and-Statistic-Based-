def getStopwords(connection):
    result = []
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM stopwords")
    for row in cursor.fetchall():
        result.append(row[1])
    cursor.close()
    return list(result)