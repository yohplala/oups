#!/usr/bin/env python3
"""
Created on Sat Dec 18 15:00:00 2021.

@author: yoh
"""
from os import path as os_path

from fastparquet import ParquetFile
from pandas import DataFrame as pDataFrame
from pandas import MultiIndex
from vaex import from_pandas

from oups.writer import to_midx
from oups.writer import write


def test_init_and_append_pandas_std(tmp_path):
    # Initialize a parquet dataset from pandas dataframe. Existing folder.
    # (no row index, compression SNAPPY, row group size: 2)
    pdf1 = pDataFrame({"a": range(6), "b": ["ah", "oh", "uh", "ih", "ai", "oi"]})
    write(str(tmp_path), pdf1, max_row_group_size=2)
    pf1 = ParquetFile(tmp_path)
    assert len(pf1.row_groups) == 3
    for rg in pf1.row_groups:
        assert rg.num_rows == 2
    res1 = pf1.to_pandas()
    assert pdf1.equals(res1)
    # Append
    pdf2 = pDataFrame({"a": range(2), "b": ["at", "of"]})
    write(str(tmp_path), pdf2, max_row_group_size=2)
    res2 = ParquetFile(tmp_path).to_pandas()
    assert pdf1.append(pdf2).reset_index(drop=True).equals(res2)


def test_init_pandas_no_folder(tmp_path):
    # Initialize a parquet dataset from pandas dataframe. No folder.
    # (no row index, compression SNAPPY, row group size: 2)
    tmp_path = os_path.join(tmp_path, "test")
    pdf = pDataFrame({"a": range(6), "b": ["ah", "oh", "uh", "ih", "ai", "oi"]})
    write(str(tmp_path), pdf, max_row_group_size=2)
    res = ParquetFile(tmp_path).to_pandas()
    assert pdf.equals(res)


def test_init_pandas_compression_brotli(tmp_path):
    # Initialize a parquet dataset from pandas dataframe. Compression 'BROTLI'.
    # (no row index)
    pdf = pDataFrame({"a": range(6), "b": ["ah", "oh", "uh", "ih", "ai", "oi"]})
    tmp_path1 = os_path.join(tmp_path, "brotli")
    write(str(tmp_path1), pdf, compression="BROTLI")
    brotli_s = os_path.getsize(os_path.join(tmp_path1, "part.0.parquet"))
    tmp_path2 = os_path.join(tmp_path, "snappy")
    write(str(tmp_path2), pdf, compression="SNAPPY")
    snappy_s = os_path.getsize(os_path.join(tmp_path2, "part.0.parquet"))
    assert brotli_s < snappy_s


def test_init_and_append_vaex_std(tmp_path):
    # Initialize a parquet dataset from vaex dataframe. Existing folder.
    # (no row index, compression SNAPPY, row group size: 2)
    pdf1 = pDataFrame({"a": range(6), "b": ["ah", "oh", "uh", "ih", "ai", "oi"]})
    vdf1 = from_pandas(pdf1)
    write(str(tmp_path), vdf1, max_row_group_size=2)
    pf1 = ParquetFile(tmp_path)
    assert len(pf1.row_groups) == 3
    for rg in pf1.row_groups:
        assert rg.num_rows == 2
    res1 = pf1.to_pandas()
    assert pdf1.equals(res1)
    # Append
    pdf2 = pDataFrame({"a": range(2), "b": ["at", "of"]})
    vdf2 = from_pandas(pdf2)
    write(str(tmp_path), vdf2, max_row_group_size=2)
    res2 = ParquetFile(tmp_path).to_pandas()
    assert pdf1.append(pdf2).reset_index(drop=True).equals(res2)


def test_init_idx_expansion_vaex(tmp_path):
    # Expanding column uindex into a 2-level column multi-index.
    pdf = pDataFrame(
        {
            "('lev1-col1','lev2-col1')": range(6, 12),
            "('lev1-col2','lev2-col2')": ["ah", "oh", "uh", "ih", "ai", "oi"],
        }
    )
    res_midx = to_midx(pdf.columns)
    ref_midx = MultiIndex.from_tuples(
        [("lev1-col1", "lev2-col1"), ("lev1-col2", "lev2-col2")], names=["l0", "l1"]
    )
    assert res_midx.equals(ref_midx)
    vdf = from_pandas(pdf)
    write(str(tmp_path), vdf, cmidx_expand=True)
    res = ParquetFile(tmp_path).to_pandas()
    pdf.columns = ref_midx
    assert res.equals(pdf)


def test_init_idx_expansion_sparse_levels(tmp_path):
    # Expanding column uindex into a 2-level column multi-index.
    pdf = pDataFrame(
        {
            "('lev1-col1','lev2-col1')": range(6, 12),
            "('lev1-col2','lev2-col2')": ["ah", "oh", "uh", "ih", "ai", "oi"],
        }
    )
    res_midx = to_midx(pdf.columns, levels=["ah"])
    ref_midx = MultiIndex.from_tuples(
        [("lev1-col1", "lev2-col1"), ("lev1-col2", "lev2-col2")], names=["ah", "l1"]
    )
    assert res_midx.equals(ref_midx)


def test_init_and_select_vaex(tmp_path):
    # Initialize a parquet dataset from vaex dataframe. Existing folder.
    # (no row index, compression SNAPPY, row group size: 2)
    pdf = pDataFrame({"a": range(6), "b": ["ah", "oh", "uh", "ih", "ai", "oi"]})
    vdf = from_pandas(pdf)
    # Select
    vdf = vdf[vdf.a > 3]
    write(str(tmp_path), vdf)
    res = ParquetFile(tmp_path).to_pandas()
    assert pdf.loc[pdf.a > 3].reset_index(drop=True).equals(res)


# Test: adding data in the middle and add new data at the end: data in the
# middle is not modified.
# Test with duplicates.
# error message crgs_ratio lower or greater than 0/1
# tester qd se met en route coalesce:
# pas assez de rows dans irgs / assez de crgs: ne se met pas en route
# se met en route si pas assez de crgs
# se met en route si assez de lignes dans irgs
# Test no duplicates but ordered_on so that new data is inserted within existing row groups, without
# drop of duplicates.

# Tester avec un cmidx l'inserting de données, est-ce que les filtres fastparquet
# marchent bien avec un cmidx?

# Tester à la fois avec pandas df & vaex df: df[0].to_numpy() & df[-1].to_numpy(): ok pour les 2?

# Tester 'ordered_on' défini (pour faire comme s'il y a 'insertion' de données)
# mais nouvelle donnée 'après' those qui existent (pas d'overlap):
# est-ce que tout se passe bien: (avec & sans coalesce, et aussi condition coalesce non remplie: la données est simplement 'append')
# test rrgs_start_idx & rrgs_end_idx bien ``None``

# Tester coalesce avec 1ers row group pas à row_group_size: correctement géré?

# Test same result between ordered_on + drop_duplicate and simple appending when there
# is no duplicate and we know data is in the end.

# Check what happens with expand_cmidx when appending data with 'update' case.
# so that it is set to the right stuff

# Check if appending of data or inserting into existing data
# If inserting into existing, and prior to rrg_start_idx défini par coalesce, do not coalesce:
