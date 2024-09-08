def getClassStatistic(connection):
    result = []
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM class_statistic")
    for row in cursor.fetchall():
        result.append({"id":row[0], "class":f"{row[1]}"})
    cursor.close()
    return list(result)