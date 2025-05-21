#!/usr/bin/env python3
"""
Created on Sun May 18 16:00:00 2025.

@author: yoh

"""
from base64 import b64decode
from base64 import b64encode
from typing import Dict

from cloudpickle import dumps
from cloudpickle import loads
from pandas import DataFrame
from pandas import MultiIndex

from oups.defines import OUPS_METADATA_KEY


def check_cmidx(cmidx: MultiIndex):
    """
    Check if column multi-index complies with fastparquet requirements.

    Library fastparquet requires names for each level in a Multiindex.
    Also, column names have to be tuple of string.

    Parameters
    ----------
    cmidx : MultiIndex
        MultiIndex to check.

    """
    # Check level names.
    if None in cmidx.names:
        raise ValueError(
            "not possible to have level name set to None.",
        )  # If an item of the column name is not a string, turn it into a string.
    # Check column names.
    for level in cmidx.levels:
        for name in level:
            if not isinstance(name, str):
                raise TypeError(f"name {name} has to be of type 'string', not '{type(name)}'.")


class ParquetAdapter:
    """
    Adapter for working either with fastparquet or arro3.
    """

    def __init__(self, use_arro3=False):
        """
        Initialize ParquetAdapter.
        """
        self.use_arro3 = use_arro3

    def write_parquet(self, path, df: DataFrame, key_value_metadata: Dict = None, **kwargs):
        """
        Write DataFrame to parquet with unified interface.
        """
        if self.use_arro3:
            from arro3.io import write_parquet

            key_value_metadata = (
                {OUPS_METADATA_KEY: b64encode(dumps(key_value_metadata)).decode()}
                if key_value_metadata
                else None
            )
            write_parquet(df, path, key_value_metadata=key_value_metadata, **kwargs)
        else:
            from fastparquet import write

            key_value_metadata = (
                {OUPS_METADATA_KEY: dumps(key_value_metadata)} if key_value_metadata else None
            )
            write(
                path,
                df,
                custom_metadata=key_value_metadata,
                file_scheme="simple",
                **kwargs,
            )

    def read_parquet(self, path, return_key_value_metadata: bool = True, **kwargs):
        """
        Read parquet and metadata with unified interface.
        """
        if self.use_arro3:
            from arro3.io import read_parquet

            table = read_parquet(path).read_all()
            df = DataFrame(table.to_struct_array().to_numpy())
            key_value_metadata = (
                loads(b64decode((table.schema.metadata_str[OUPS_METADATA_KEY]).encode()))
                if return_key_value_metadata
                else None
            )
        else:
            from fastparquet import ParquetFile

            pf = ParquetFile(path)
            df = pf.to_pandas()
            key_value_metadata = (
                loads(pf.key_value_metadata[OUPS_METADATA_KEY])
                if return_key_value_metadata
                else None
            )

        return (df, key_value_metadata) if return_key_value_metadata else df
