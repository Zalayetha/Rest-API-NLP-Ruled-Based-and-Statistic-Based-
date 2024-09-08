def getClassStatistic(connection):
    result = []
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM class_statistic")
    for row in cursor.fetchall():
        result.append({"id":row[0], "class":f"{row[1]}"})
    cursor.close()
    return list(result)

def getClassStatisticById(id, connection):
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM class_statistic WHERE id = {id}")
    for row in cursor.fetchall():
        result = {"id":row[0], "class":f"{row[1]}"}
    cursor.close()
    return result