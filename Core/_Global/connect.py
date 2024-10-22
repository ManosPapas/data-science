from standard import *
from connection import *
from constants import *


def db_connect(db_name=DB_1):
    """Establishes a connection to the specified database."""
    return db_connection(db_connection_string(db_name))


def db_connection_string(db_name=DB_1):
    """Constructs the connection string for the specified database."""
    server = CONFIG[db_name]["SERVER"]
    database = CONFIG[db_name]["DATABASE"]
    username = CONFIG[db_name]["USER_NAME"]
    password = CONFIG[db_name]["PASSWORD"]
    driver = CONFIG[db_name]["DRIVER"]

    return f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"


def db_connection(db_connection_string):
    """Creates and returns a database connection using the provided connection string."""
    try:
        connection = pyodbc.connect(db_connection_string)
        return connection
    except Exception as e:
        raise Exception(f"Database connection failed: {e}")


def db_close(connection):
    """Closes the specified database connection if it is not None."""
    if connection is not None:
        connection.close()


def execute_sql_raw(sql, db_name=DB_1, read=True):
    """Executes a raw SQL query on the specified database and returns the result."""
    connection = db_connect(db_name)
    if read:
        result = pd.read_sql(sql, connection)
    else:
        result = connection.cursor().execute(sql)
        connection.commit()

    db_close(connection)

    return result


def execute_sql(file_path, db_name=DB_1, sql_params=None):
    """Executes SQL queries from a file."""
    sql = read_file(file_path)

    if sql_params:
        for key, value in sql_params.items():
            placeholder = "{" + key + "}"
            sql = sql.replace(placeholder, str(value))

    return execute_sql_raw(sql, db_name)


def read_file(file_path):
    """Reads the contents of the specified SQL file and returns it as a string."""
    file_path = SQL_PATH + file_path + ".sql"
    file = open(file_path, "r")
    data = file.read()
    file.close()

    return data


def db_to_sql(df, table, connection=DB_3, action="append", index_=False):
    """Writes a DataFrame to a specified SQL table."""
    engine = create_engine(
        "mssql+pyodbc:///?odbc_connect=" + db_connection_string(connection)
    )
    df.to_sql(table, engine, if_exists=action, index=index_)
