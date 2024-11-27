from fastparquet import ParquetFile
from fastparquet import write
from pandas import DataFrame


def create_parquet_file(tmp_path: str, df: DataFrame, row_group_offsets: int) -> ParquetFile:
    """
    Create a temporary parquet file for testing.

    Parameters
    ----------
    tmp_path : str
        Temporary directory provided by pytest.
    df : DataFrame
        Data to write to the parquet file.
    row_group_offsets : int
        Number of rows per row group.

    Returns
    -------
    ParquetFile
        The created parquet file object.

    Notes
    -----
    The file is created using the 'hive' file scheme and stored in a directory
    named 'test_parquet' within the temporary directory.

    """
    path = f"{tmp_path}/test_parquet"
    write(path, df, row_group_offsets=row_group_offsets, file_scheme="hive")
    return ParquetFile(path)
