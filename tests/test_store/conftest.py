import pytest
from fastparquet import ParquetFile
from fastparquet import write
from pandas import DataFrame


@pytest.fixture
def create_parquet_file(tmp_path):
    """
    Create a temporary parquet file for testing.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest.

    Returns
    -------
    callable
        Function that creates a ParquetFile with specified row group size.

    """

    def _create_parquet(df: DataFrame, row_group_offsets: int) -> ParquetFile:
        path = f"{tmp_path}/test.parquet"
        write(path, df, row_group_offsets=row_group_offsets, file_scheme="hive")
        return ParquetFile(path)

    return _create_parquet
