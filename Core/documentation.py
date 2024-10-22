##############################################################################################################################
##############################################################################################################################

## !!!An exampele of documentation!!!

doc_strings__doc__ =    """
    Retrieves the documentation string (docstring) of a given function from the 'documentation' module.

    This function attempts to find the function in the 'documentation' module by the given name. If it
    exists and has a docstring, it returns the docstring. Otherwise, it provides an appropriate message
    indicating that either the function does not exist or documentation is not available.

    Parameters:
        - function_name (str): The name of the function for which to retrieve the docstring.

    Returns:
        - If the function exists and has a docstring, it returns the docstring as a string.
        - If the function does not have a docstring, it returns a string saying 'No documentation available for <function_name>.'
        - If the function does not exist in the 'documentation' module, it returns a string saying '<function_name> does not exist in the specified module.'

    Raises:
        AttributeError: If the 'documentation' module does not exist or 'getattr' fails to retrieve the function.
    """

## _Global

# connect.py

db_connect__doc__ =     """
    def db_connect(connection='DB_1'):
    Connects to a predefined database based on the provided connection name.
    Defaults to connecting to 'DB_1' database.

    Parameters:
        - db_name: The name of the predefined database connection (e.g., 'DB_1' or 'DB_2').

    Returns:
        - The database connection object if the connection is successful.
        - Passes on the exception if the db_connection function encounters an issue.

    Example:
        - db_connect('DB_1') # Connects to the 'DB_1' database.
        - db_connect('DB_2') # Connects to the 'DB_2' database.
    """


db_connection_string__doc__=    """
    Creates the parameter string for connecting to the given db name,

    Parameters:
        - db_name (DatabaseConnection): The name of the database.

    Returns:
        - the connection string for the given db.
    """


db_connection__doc__ =         """
    def db_connection(server, database, username, password, driver='{SQL Server}'):
    Establish a connection to the database using the provided parameters.

    Parameters:
        - db_connection_string: the connection string of the db.

    Returns:
        - The database connection object if the connection is successful.
        - Raises an exception if the connection fails.
    """


db_close__doc__ =     """
    Closes the database connection provided as an argument.

    Parameters:
        - connection (DatabaseConnection): An open database connection object. 
            This object should have a `.close()` method which handles the process of 
            disconnecting from the database.

    Raises:
        - AttributeError: If the connection object does not have a `.close()` method.
    """


execute_sql_raw__doc__ =     """
    Execute a raw SQL query on a specified database.

    Parameters:
        - sql : str
            - The SQL query to be executed.
        - db_name : str, optional
            - The name of the database to connect to (default is 'DB_1').
        - read: boolean
            - If the SQL statement is for reading then true, otherwise false.
        
    Returns:
        - pd.DataFrame
            - A DataFrame containing the results of the query.

    Raises:
        - Exception
            An exception is raised if there is an error connecting to the database or executing the query.
    """


execute_sql__doc__ =     """Executes an SQL script from a specified file against a database.

    Params:
        - file_name (str): The name of the SQL file (without extension) to be executed.
        - db_name (str): The name of the database to execute the script against. Defaults to 'DB_1'.
        - sql_params (dictionary): The params you want to pass to your SQL query.
    Returns:
        - Any: The result of the SQL execution.
    """


read_file__doc__ =     """
    Reads the content of a file.

    Params:
        - file_path: Path to the file to be read.
    Returns: 
        - The file content as a string.
    """


exchange_rates__doc__ =     """
    Get exchange rates for a given currency against a list of global currencies.

    Params:
        - param currency: The base currency code to get rates for.
        - param remove_current: Flag to remove the current currency from the list if True.
    
    Returns: 
        - A dictionary of exchange rates.
    """


db_connect__doc__ =    """
    Retrieves the documentation string (docstring) of a given function from the 'documentation' module.

    This function attempts to find the function in the 'documentation' module by the given name. If it
    exists and has a docstring, it returns the docstring. Otherwise, it provides an appropriate message
    indicating that either the function does not exist or documentation is not available.

    Parameters:
        - function_name (str): The name of the function for which to retrieve the docstring.

    Returns:
        - If the function exists and has a docstring, it returns the docstring as a string.
        - If the function does not have a docstring, it returns a string saying 'No documentation available for <function_name>.'
        - If the function does not exist in the 'documentation' module, it returns a string saying '<function_name> does not exist in the specified module.'

    Raises:
        - AttributeError: If the 'documentation' module does not exist or 'getattr' fails to retrieve the function.
    """


db_to_sql__doc__ =    """
    Write the DataFrame to a SQL database table.

    Parameters:
        - df: pandas.DataFrame
        The DataFrame to write to SQL.
        - table: str
        The name of the target SQL table.
        - connection: str, optional
        The connection name which refers to specific database connection settings.
        Default is 'Analytics'.
        - action: str, {'fail', 'replace', 'append'}, default 'append'
        How to behave if the table already exists:
        - fail: Raise a ValueError.
        - replace: Drop the table before inserting new values.
        - append: Insert new values to the existing table.
        - index_: bool, optional
        Write DataFrame index as a column. Default is False.

    Returns:
        - None
    """




##############################################################################################################################
##############################################################################################################################
