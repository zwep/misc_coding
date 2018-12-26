
import os
import MySQLdb
import config

seb_mysql_key = os.environ['seb_mysql_key']
conn = MySQLdb.connect('localhost', 'seb', seb_mysql_key)
cursor = conn.cursor()

# Show databases
databases = ("show databases")
cursor.execute(databases)
res_databases = cursor.fetchall()

# Show tables
for i_db in res_databases:
    cursor.execute("use " + i_db[0])
    cursor.execute("show tables")
    res_tables = cursor.fetchall()
    print(i_db, res_tables)