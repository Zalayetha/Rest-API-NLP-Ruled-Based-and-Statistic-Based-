def getNormalization(connection):
    result = []
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM normalization")
    for row in cursor.fetchall():
        result.append({f"{row[1]}":f"{row[2]}"})
    cursor.close()
    return list(result)