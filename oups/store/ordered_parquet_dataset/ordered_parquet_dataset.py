#!/usr/bin/env python3
"""
Created on Wed Dec 26 22:30:00 2021.

@author: yoh

"""
from pathlib import Path
from typing import List

from numpy import iinfo
from pandas import DataFrame

from oups.defines import KEY_FILE_IDS
from oups.defines import KEY_ORDERED_ON
from oups.store.ordered_parquet_dataset.metadata_filename import get_md_filepath
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset_base import RGS_STATS_BASE_DTYPES
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset_base import RGS_STATS_COLUMNS
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset_base import (
    BaseOrderedParquetDataset,
)
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset_base import get_parquet_filepaths
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset_base import parquet_adapter


class OrderedParquetDataset(BaseOrderedParquetDataset):
    """
    Ordered Parquet Dataset.

    This class inherits from BaseOrderedParquetDataset and provides the full
    implementation with modification capabilities. The __getitem__ method
    returns read-only views for safe data access.

    All functionality is inherited from BaseOrderedParquetDataset.

    Notes
    -----
    - There is one row group per file.
    - Dataset metadata are written in a separate file in parquet format, located
      at the same level than the dataset directory (not within the directory).
      This way, if provided the directory path, another parquet reader can read
      the dataset without being confused by this metadata file.
    - File ids (in file names) have the same number of digits. This is to ensure
      that files can be read in the correct order by other parquet readers.

    """

    # All methods inherited from BaseOrderedParquetDataset
    pass


def create_custom_opd(tmp_path: str, df: DataFrame, row_group_offsets: List[int], ordered_on: str):
    """
    Create a custom opd for testing.

    Parameters
    ----------
    tmp_path : str
        Temporary directory wheere to locate the opd files.
    df : DataFrame
        Data to write to opd files.
    row_group_offsets : List[int]
        Start index of row groups in 'df'.
    ordered_on : str
        Column name to order row groups by.


    Returns
    -------
    OrderedParquetDataset
        The created opd object.

    """
    _max_allowed_file_id = iinfo(RGS_STATS_BASE_DTYPES[KEY_FILE_IDS]).max
    _file_id_n_digits = len(str(_max_allowed_file_id))
    n_rows = []
    ordered_on_mins = []
    ordered_on_maxs = []
    row_group_ends_excluded = row_group_offsets[1:] + [len(df)]
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    for file_id, (row_group_start, row_group_end_excluded) in enumerate(
        zip(row_group_offsets, row_group_ends_excluded),
    ):
        df_rg = df.iloc[row_group_start:row_group_end_excluded]
        n_rows.append(len(df_rg))
        ordered_on_mins.append(df_rg.loc[:, ordered_on].iloc[0])
        ordered_on_maxs.append(df_rg.loc[:, ordered_on].iloc[-1])
        parquet_adapter.write_parquet(
            path=get_parquet_filepaths(tmp_path, file_id, _file_id_n_digits),
            df=df_rg,
            # file_scheme="simple",   # not needed, is already a parameter in parquet_adapter.write_parquet()
        )
    row_group_stats = DataFrame(
        data=zip(range(len(row_group_offsets)), n_rows, ordered_on_mins, ordered_on_maxs),
        columns=RGS_STATS_COLUMNS,
    ).astype(RGS_STATS_BASE_DTYPES)
    parquet_adapter.write_parquet(
        path=get_md_filepath(tmp_path),
        df=row_group_stats,
        # file_scheme="simple",   # not needed, is already a parameter in parquet_adapter.write_parquet()
        key_value_metadata={KEY_ORDERED_ON: ordered_on},
    )
    return OrderedParquetDataset(tmp_path)
