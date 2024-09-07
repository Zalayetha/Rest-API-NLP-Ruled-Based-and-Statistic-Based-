from flask import current_app, g
import mysql.connector

def getdb():
    if 'db' not in g or not g.db.is_connected():
        g.db = mysql.connector.connect(
            host='127.0.0.1',
            port='8889',
            user='root',
            password='root',
            database='SANTANA',
        )
    return g.db

def close_db(e=None):
    db = g.pop('db', None)

    if db is not None and db.is_connected():
        db.close()

    