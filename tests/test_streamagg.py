#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh
"""

# from fastparquet import ParquetFile
# from fastparquet import write as fp_write
# from numpy import arange
# from pandas import DataFrame as pDataFrame
# from pandas import MultiIndex
# from pandas import concat
# from vaex import from_pandas

# from oups.streamagg import streamagg


# tmp_path = os_path.expanduser('~/Documents/code/data/oups')

# WiP
# test with ParquetFile seed
# test with vaex seed

# Test ValueError when not discard_last and not last_seed_index in seed metadata.

# discard_last : seed_index_end correctly taken into account?

# test when restarting with existing aggregated data, seed_index_start correctly taken into account?

# from seed data, test with additional columns and check only cols_in to be used are
# indeed loaded.

# Test with 'by' as callable, with 'buffer_binning' and without 'buffer_binning'.

# Test avec streamagg 1 se terminant exactement sure la bin en cours, & streamagg 2 reprenant sure une nouvelle bin
# Et quand streamagg est utile. (itération 2 démarrée au milieu de bin 1 par example)

# test with a single new rowin seed data.


# Test error bin_on not defined, but 'by' is a callable

# Test error message if 'bin_on' is already used as an output column name from aggregation

# test case when one aggregation chunk is a single line and is not agged with next aggregation result (for instance
# in row groups of seed data, a single bin / single row, and next row group of seed data is a new bin)

# test with "last_complete_seed_index": use a streamagg result within the store
# does jsonification work with a Timestamp?
