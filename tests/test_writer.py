#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 15:00:00 2021
@author: yoh
"""
from os import path as os_path
import pytest

from pandas import DataFrame as pDataFrame, MultiIndex
from fastparquet import ParquetFile
from vaex import from_pandas

from oups.collection import CMIDX_SEP
from oups.writer import write, to_midx


def test_init_and_append_pandas_std(tmp_path):
    # Initialize a parquet dataset from pandas dataframe. Existing folder.
    # (no row index, compression SNAPPY, row group size: 2)
    pdf1 = pDataFrame({'a':range(6), 'b':['ah','oh','uh','ih','ai','oi']})
    write(str(tmp_path), pdf1, row_group_size=2)
    pf1 = ParquetFile(tmp_path)
    assert len(pf1.row_groups) == 3
    for rg in pf1.row_groups:
        assert rg.num_rows == 2
    res1 = pf1.to_pandas()
    assert pdf1.equals(res1)
    # Append
    pdf2 = pDataFrame({'a':range(2), 'b':['at','ot']})
    write(str(tmp_path), pdf2, row_group_size=2)
    res2 = ParquetFile(tmp_path).to_pandas()
    assert pdf1.append(pdf2).reset_index(drop=True).equals(res2)

def test_init_pandas_no_folder(tmp_path):
    # Initialize a parquet dataset from pandas dataframe. No folder.
    # (no row index, compression SNAPPY, row group size: 2)
    tmp_path = os_path.join(tmp_path,'test')
    pdf = pDataFrame({'a':range(6), 'b':['ah','oh','uh','ih','ai','oi']})
    write(str(tmp_path), pdf, row_group_size=2)
    res = ParquetFile(tmp_path).to_pandas()
    assert pdf.equals(res)   

def test_init_pandas_compression_brotli(tmp_path):
    # Initialize a parquet dataset from pandas dataframe. Compression 'BROTLI'.
    # (no row index)
    pdf = pDataFrame({'a':range(6), 'b':['ah','oh','uh','ih','ai','oi']})
    tmp_path1 = os_path.join(tmp_path,'brotli')
    write(str(tmp_path1), pdf, compression='BROTLI')
    brotli_s = os_path.getsize(os_path.join(tmp_path1,'part.0.parquet'))
    tmp_path2 = os_path.join(tmp_path,'snappy')
    write(str(tmp_path2), pdf, compression='SNAPPY')
    snappy_s = os_path.getsize(os_path.join(tmp_path2,'part.0.parquet'))
    assert brotli_s < snappy_s

def test_init_and_append_vaex_std(tmp_path):
    # Initialize a parquet dataset from vaex dataframe. Existing folder.
    # (no row index, compression SNAPPY, row group size: 2)
    pdf1 = pDataFrame({'a':range(6), 'b':['ah','oh','uh','ih','ai','oi']})
    vdf1 = from_pandas(pdf1)
    write(str(tmp_path), vdf1, row_group_size=2)
    pf1 = ParquetFile(tmp_path)
    assert len(pf1.row_groups) == 3
    for rg in pf1.row_groups:
        assert rg.num_rows == 2
    res1 = pf1.to_pandas()
    assert pdf1.equals(res1)
    # Append
    pdf2 = pDataFrame({'a':range(2), 'b':['at','ot']})
    vdf2 = from_pandas(pdf2)
    write(str(tmp_path), vdf2, row_group_size=2)
    res2 = ParquetFile(tmp_path).to_pandas()
    assert pdf1.append(pdf2).reset_index(drop=True).equals(res2)

def test_init_idx_expansion_vaex(tmp_path):
    # Expanding column uindex into a 2-level column multi-index.
    # ('level_sep' set to '__')
    level_sep = CMIDX_SEP
    pdf = pDataFrame({f'lev1-col1{level_sep}lev2-col1':range(6,12),
                      f'lev1-col2{level_sep}lev2-col2':
                                              ['ah','oh','uh','ih','ai','oi']})
    res_midx = to_midx(pdf.columns, level_sep)
    ref_midx = MultiIndex.from_tuples([('lev1-col1','lev2-col1'),
                                       ('lev1-col2','lev2-col2')],
                                      names=['l0', 'l1'])
    assert res_midx.equals(ref_midx)
    vdf = from_pandas(pdf)
    write(str(tmp_path), vdf, cmidx_expand=True, cmidx_sep=level_sep)
    res = ParquetFile(tmp_path).to_pandas()
    pdf.columns = ref_midx
    assert res.equals(pdf)

def test_init_idx_expansion_sparse_levels(tmp_path):
    # Expanding column uindex into a 2-level column multi-index.
    # ('level_sep' set to '__')
    level_sep = CMIDX_SEP
    pdf = pDataFrame({f'lev1-col1{level_sep}lev2-col1':range(6,12),
                      f'lev1-col2{level_sep}lev2-col2':
                                              ['ah','oh','uh','ih','ai','oi']})
    res_midx = to_midx(pdf.columns, level_sep, levels=['ah'])
    ref_midx = MultiIndex.from_tuples([('lev1-col1','lev2-col1'),
                                       ('lev1-col2','lev2-col2')],
                                      names=['ah', 'l1'])
    assert res_midx.equals(ref_midx)

def test_init_and_select_vaex(tmp_path):
    # Initialize a parquet dataset from vaex dataframe. Existing folder.
    # (no row index, compression SNAPPY, row group size: 2)
    pdf = pDataFrame({'a':range(6), 'b':['ah','oh','uh','ih','ai','oi']})
    vdf = from_pandas(pdf)
    # Select
    vdf = vdf[vdf.a>3]
    write(str(tmp_path), vdf)
    res = ParquetFile(tmp_path).to_pandas()
    assert pdf.loc[pdf.a >3].reset_index(drop=True).equals(res)

def test_cmidx_exception(tmp_path):
    # Exception in case 'cmidx_expand' is `True`, but 'cmidx_sep' is not set.
    level_sep = CMIDX_SEP
    pdf = pDataFrame({f'lev1-col1{level_sep}lev2-col1':range(6,12),
                      f'lev1-col2{level_sep}lev2-col2':
                                              ['ah','oh','uh','ih','ai','oi']})
    with pytest.raises(ValueError, match="^Setting `cmidx`"):
        write(str(tmp_path), pdf, cmidx_expand=True)
