from standard import *
import documentation


def doc_strings(function_name):
    """Retrieves and prints the documentation for the specified function."""
    try:
        function_name = f"{function_name}__doc__"
        obj = getattr(documentation, function_name)

        if obj:
            print(obj)
        else:
            return f"No documentation available for {function_name}."
    except AttributeError:
        return f"{function_name} does not exist in the specified module."


def print_dataframe_shape(df):
    """Prints the shape (dimensions) of the given DataFrame."""
    print(f"Dimensions: {df.shape}")


def print_info(df):
    """Prints the shape and columns of the given DataFrame."""
    print(f"Dimensions: {df.shape}\n")
    print(f"Columns: {list(df.columns)}")


def export_csv(df, name, path=""):
    """Exports the given DataFrame to a CSV file at the specified path."""
    if not isinstance(df, pd.DataFrame):
        print("The data provided is not a pandas DataFrame.")
        return

    if not name:
        print("Please provide a valid file name.")
        return

    path = default_export_path(path)
    
    # Ensure the directory exists; if not, create it.
    os.makedirs(path, exist_ok=True)
    
    full_path = os.path.join(path, f"{name}.csv")
    
    try:
        df.to_csv(full_path, index=False)
    except Exception as e:
        print(f"Failed to export the CSV file: {e}")
        return
    
    print(f"Export is completed for CSV file: {name}.csv at {path}")


def export_excel(df_list, sheet_list, filename, path="", engine="openpyxl"):
    """Exports a list of DataFrames to an Excel file with specified sheets."""
    path = default_export_path(path)

    m = 1_000_000
    
    with pd.ExcelWriter(os.path.join(path, f"{filename}.xlsx"), engine=engine) as writer:
        for index, df in enumerate(df_list):
            sheet_name = sheet_list[index]
            
            row = 0
            for start in range(0, df.shape[0], m):
                end = min(start + m, df.shape[0])
                df.iloc[start:end].to_excel(writer, sheet_name=sheet_name, startrow=row, index=False)
                row += (end - start)

    print(f"Export is completed for Excel file: {filename}.xlsx")


def print_dataset(df):
    """Prints the entire DataFrame as a string."""
    print(df.to_string())


def default_export_path(path):
    """Returns the default export path if none is specified."""
    return EXPORT_PATH if not path else path


def parse_dates_(df, date_columns):
    """Parses specified date columns in the DataFrame to datetime format."""
    for col in date_columns:
        df[col] = df[col].apply(np.datetime64)

    return df


def parallel_run(tasks):
    """Executes a list of tasks in parallel and returns their results."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(task) for task in tasks]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    return results
