from classStatistic.classStatistic import getClassStatistic,getClassStatisticById
def insertPreprocessingStatistic(connection,dataframe):
    print("Insert Preprocessing Statistic....")
    print("data : ",dataframe)
    cursor = connection.cursor()

    # get data preprcocesing statistic
    data_preprocessing = getPreprocessingStatistic(connection=connection)
    
    if len(data_preprocessing) > 0:
        deleteAllRecordPreprocessingStatistic(connection=connection)
        
    # Translate classs statistic
    data = translateClassStatistic(connection=connection,data=dataframe)

    # set the default value when class is None
    data.fillna(9, inplace=True)
    # Insert dataframe into MySQL table
    for index, row in data.iterrows():
        sql = "INSERT INTO preprocessing_statistic (currentword, currenttag, bef1tag, tokentype, id_class) VALUES (%s, %s, %s, %s, %s)"
        values = (row['currentword'], row['currenttag'], row['bef1tag'], row['token'], row['class'])
        
        cursor.execute(sql, values)

    # Commit the transaction
    connection.commit()

    # Close the connection
    cursor.close()
    connection.close()

# Function to search id class by the class name
def findIdByClass(className, data):
    for item in data:
        if item['class'] == className:
            return item['id']

    return None

# Function to translate class statistic to numeric based on id in database
def translateClassStatistic(connection,data):
    print("Translate Class Statistic....")
    temp_df = data

    response = getClassStatistic(connection=connection)
    # get the class in dataframe
    set_class = set(temp_df["class"])

    for cls in set_class:
        temp_df.loc[temp_df["class"] == cls,
                    "class"] = findIdByClass(cls, response)

    print(temp_df)
    print("Translate Class Statistic Done....")
    return temp_df


def getPreprocessingStatistic(connection):
    result = []
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM preprocessing_statistic")
    for row in cursor.fetchall():
        class_name = getClassStatisticById(id=int(row[5]),connection=connection)['class']
        result.append({
            "currentword":row[1],
            "currenttag":row[3],
            "bef1tag":row[4],
            "tokentype":row[2],
            "class":class_name,
        })
    cursor.close()
    return list(result)


def deleteAllRecordPreprocessingStatistic(connection):
    cursor = connection.cursor()
    cursor.execute("DELETE FROM preprocessing_statistic")
    connection.commit()
    cursor.close()
