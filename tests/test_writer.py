#!/usr/bin/env python3
"""
Created on Sat Dec 18 15:00:00 2021.

@author: yoh
"""
from os import path as os_path

from fastparquet import ParquetFile
from fastparquet import write as fp_write
from numpy import arange
from pandas import DataFrame as pDataFrame
from pandas import MultiIndex
from vaex import from_pandas

from oups.writer import to_midx
from oups.writer import write as ps_write


def test_init_and_append_pandas_std(tmp_path):
    # Initialize a parquet dataset from pandas dataframe. Existing folder.
    # (no row index, compression SNAPPY, row group size: 2)
    pdf1 = pDataFrame({"a": range(6), "b": ["ah", "oh", "uh", "ih", "ai", "oi"]})
    ps_write(str(tmp_path), pdf1, max_row_group_size=2)
    pf1 = ParquetFile(tmp_path)
    assert len(pf1.row_groups) == 3
    for rg in pf1.row_groups:
        assert rg.num_rows == 2
    res1 = pf1.to_pandas()
    assert pdf1.equals(res1)
    # Append
    pdf2 = pDataFrame({"a": range(2), "b": ["at", "of"]})
    ps_write(str(tmp_path), pdf2, max_row_group_size=2)
    res2 = ParquetFile(tmp_path).to_pandas()
    assert pdf1.append(pdf2).reset_index(drop=True).equals(res2)


def test_init_pandas_no_folder(tmp_path):
    # Initialize a parquet dataset from pandas dataframe. No folder.
    # (no row index, compression SNAPPY, row group size: 2)
    tmp_path = os_path.join(tmp_path, "test")
    pdf = pDataFrame({"a": range(6), "b": ["ah", "oh", "uh", "ih", "ai", "oi"]})
    ps_write(str(tmp_path), pdf, max_row_group_size=2)
    res = ParquetFile(tmp_path).to_pandas()
    assert pdf.equals(res)


def test_init_pandas_compression_brotli(tmp_path):
    # Initialize a parquet dataset from pandas dataframe. Compression 'BROTLI'.
    # (no row index)
    pdf = pDataFrame({"a": range(6), "b": ["ah", "oh", "uh", "ih", "ai", "oi"]})
    tmp_path1 = os_path.join(tmp_path, "brotli")
    ps_write(str(tmp_path1), pdf, compression="BROTLI")
    brotli_s = os_path.getsize(os_path.join(tmp_path1, "part.0.parquet"))
    tmp_path2 = os_path.join(tmp_path, "snappy")
    ps_write(str(tmp_path2), pdf, compression="SNAPPY")
    snappy_s = os_path.getsize(os_path.join(tmp_path2, "part.0.parquet"))
    assert brotli_s < snappy_s


def test_init_and_append_vaex_std(tmp_path):
    # Initialize a parquet dataset from vaex dataframe. Existing folder.
    # (no row index, compression SNAPPY, row group size: 2)
    pdf1 = pDataFrame({"a": range(6), "b": ["ah", "oh", "uh", "ih", "ai", "oi"]})
    vdf1 = from_pandas(pdf1)
    ps_write(str(tmp_path), vdf1, max_row_group_size=2)
    pf1 = ParquetFile(tmp_path)
    assert len(pf1.row_groups) == 3
    for rg in pf1.row_groups:
        assert rg.num_rows == 2
    res1 = pf1.to_pandas()
    assert pdf1.equals(res1)
    # Append
    pdf2 = pDataFrame({"a": range(2), "b": ["at", "of"]})
    vdf2 = from_pandas(pdf2)
    ps_write(str(tmp_path), vdf2, max_row_group_size=2)
    res2 = ParquetFile(tmp_path).to_pandas()
    assert pdf1.append(pdf2).reset_index(drop=True).equals(res2)


def test_init_idx_expansion_vaex(tmp_path):
    # Expanding column index into a 2-level column multi-index.
    # No level names.
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
    ps_write(str(tmp_path), vdf, cmidx_expand=True)
    res = ParquetFile(tmp_path).to_pandas()
    pdf.columns = ref_midx
    assert res.equals(pdf)


def test_init_idx_expansion_sparse_levels(tmp_path):
    # Expanding column index into a 2-level column multi-index.
    # Sparse level names.
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
    ps_write(str(tmp_path), vdf)
    res = ParquetFile(tmp_path).to_pandas()
    assert pdf.loc[pdf.a > 3].reset_index(drop=True).equals(res)


def test_pandas_coalescing_simple_irgs(tmp_path):
    # Initialize a parquet dataset directly with fastparquet.
    # max_row_group_size = 4
    # (incomplete row group size: 1 to be 'incomplete')
    # irgs_max = 2

    # Case 1, 'irgs_max" not reached yet.
    # (size of new data: 1)
    # One incomplete row group in the middle of otherwise complete row groups.
    # Because there is only 1 irgs (while max is 2), and 2 rows over all irgs
    # (including data to be written), coalesing is not activated.
    # rgs                          [ 0,  ,  ,  , 1, 2,  ,  ,  , 3]
    # idx                          [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # a                            [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # a (new data)                                               [20]
    # rgs (new)                    [ 0,  ,  ,  , 1, 2,  ,  ,  , 3,4 ]
    pdf1 = pDataFrame({"a": range(10)})
    dn = os_path.join(tmp_path, "test")
    fp_write(dn, pdf1, row_group_offsets=[0, 4, 5, 9], file_scheme="hive", write_index=False)
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [4, 1, 4, 1]
    max_row_group_size = 4
    irgs_max = 2
    pdf2 = pDataFrame({"a": [20]})
    ps_write(dn, pdf2, max_row_group_size=max_row_group_size, irgs_max=irgs_max)
    pf_rec1 = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf_rec1.row_groups]
    assert len_rgs == [4, 1, 4, 1, 1]
    df_ref1 = pdf1.append({"a": 20}, ignore_index=True)
    assert pf_rec1.to_pandas().equals(df_ref1)

    # Case 2, 'irgs_max" now reached.
    # rgs                          [ 0,  ,  ,  , 1, 2,  ,  ,  , 3, 4]
    # idx                          [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10]
    # a                            [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,20]
    # a (new data)                                                  [20]
    # rgs (new)                    [ 0,  ,  ,  , 1, 2,  ,  ,  , 3,  ,  ]
    ps_write(dn, pdf2, max_row_group_size=max_row_group_size, irgs_max=irgs_max)
    pf_rec2 = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf_rec2.row_groups]
    assert len_rgs == [4, 1, 4, 3]
    df_ref2 = df_ref1.append({"a": 20}, ignore_index=True)
    assert pf_rec2.to_pandas().equals(df_ref2)


def test_pandas_coalescing_simple_max_row_group_size(tmp_path):
    # Initialize a parquet dataset directly with fastparquet.
    # max_row_group_size = 4
    # (incomplete row group size: 1 to be 'incomplete')
    # irgs_max = 5
    # Coalescing occurs because 'max_row_group_size' is reached.
    # In initial dataset, there are 3 row groups with a single row.
    # rgs                          [ 0,  ,  ,  , 1, 2,  ,  ,  , 3, 4, 5]
    # idx                          [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11]
    # a                            [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11]
    # a (new data)                                                     [20]
    # rgs (new)                    [ 0,  ,  ,  , 1, 2,  ,  ,  , 3,  ,  ,  ]
    pdf1 = pDataFrame({"a": range(12)})
    dn = os_path.join(tmp_path, "test")
    fp_write(
        dn, pdf1, row_group_offsets=[0, 4, 5, 9, 10, 11], file_scheme="hive", write_index=False
    )
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [4, 1, 4, 1, 1, 1]
    max_row_group_size = 4
    irgs_max = 5
    # With additional row of new data, 'max_row_group_size' is reached.
    pdf2 = pDataFrame({"a": [20]})
    ps_write(dn, pdf2, max_row_group_size=max_row_group_size, irgs_max=irgs_max)
    pf_rec1 = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf_rec1.row_groups]
    assert len_rgs == [4, 1, 4, 4]
    df_ref1 = pdf1.append({"a": 20}, ignore_index=True)
    assert pf_rec1.to_pandas().equals(df_ref1)


def test_pandas_appending_data_with_drop_duplicates(tmp_path):
    # rgs                          [ 0,  ,  ,  , 1, 2,  ,  ,  , 3, 4]
    # idx                          [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10]
    # a                            [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10]
    # b                            [10,11,12,13,14,15,16,17,18,19,20]
    # a (new data, ordered_on, duplicates_on)                       [10,20]
    # b (new data, check last)                                      [11,31]
    # 1 duplicate                                                  x  x
    # rgs (new)                    [ 0,  ,  ,  , 1, 2,  ,  ,  , 3, x, 4,  ]
    pdf1 = pDataFrame({"a": range(11), "b": range(10, 21)})
    dn = os_path.join(tmp_path, "test")
    fp_write(dn, pdf1, row_group_offsets=[0, 4, 5, 9, 10], file_scheme="hive", write_index=False)
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [4, 1, 4, 1, 1]
    max_row_group_size = 4
    pdf2 = pDataFrame({"a": [10, 20], "b": [11, 31]})
    # Dropping duplicates '10', 'ordered_on' with 'a'.
    ps_write(dn, pdf2, max_row_group_size=max_row_group_size, ordered_on="a", duplicates_on="a")
    pf_rec = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf_rec.row_groups]
    # 'coalesce' mode not requested, so no end row group merging.
    assert len_rgs == [4, 1, 4, 1, 2]
    df_ref = pdf1.append({"a": 20, "b": 31}, ignore_index=True)
    df_ref.iloc[10] = [10, 11]
    assert pf_rec.to_pandas().equals(df_ref)


def test_pandas_appending_data_with_sharp_starts(tmp_path):
    # Validate:
    # - 'sharp starts' (meaning that bins splitting the dataframe to be written
    #   are adjusted so as not to fall in the middle of duplicates.)
    # - is also tested the index 'a' being added to 'duplicates_on', as the 2
    #   last values in 'pdf2' are not dropped despite being duplicates on 'b'.
    # - drop duplicates, keep 'last.
    # rgs                  [0, , ,1, , ,2, ]
    # idx                  [0, , ,3, , ,6, ]
    # a                    [0,1,2,3,3,3,4,5]
    # b                    [0, , ,3, , ,6, ]
    # c                    [0,0,0,0,0,0,0,0]
    # a (new data, ordered_on)             [5,5,5,5,6,6, 7, 8]
    # b (new data, duplicates_on)          [7,7,8,9,9,9,10,10]
    # c (new data, check last)             [1,2,3,4,5,6, 7, 8]
    # 3 duplicates (on b)                 x x x,  x x x, x  x
    # rgs (new)            [0, , ,1, , ,2,x,x,3, , ,x,4,  , 5]
    # idx                  [0, , ,3, , ,6,    7, , , 10,  ,  ]
    a1 = [0, 1, 2, 3, 3, 3, 4, 5]
    len_a1 = len(a1)
    b1 = range(len_a1)
    c1 = [0] * len_a1
    pdf1 = pDataFrame({"a": a1, "b": b1, "c": c1})
    dn = os_path.join(tmp_path, "test")
    fp_write(dn, pdf1, row_group_offsets=[0, 3, 6], file_scheme="hive", write_index=False)
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [3, 3, 2]
    max_row_group_size = 3
    a2 = [5, 5, 5, 5, 6, 6, 7, 8]
    len_a2 = len(a2)
    b2 = [7, 7, 8, 9, 9, 9, 10, 10]
    c2 = arange(len_a2) + 1
    pdf2 = pDataFrame({"a": a2, "b": b2, "c": c2})
    # 'ordered_on' with 'a', duplicates on 'b' ('a' added implicitly)
    ps_write(dn, pdf2, max_row_group_size=max_row_group_size, ordered_on="a", duplicates_on="b")
    pf_rec = ParquetFile(dn)
    len_rgs_rec = [rg.num_rows for rg in pf_rec.row_groups]
    assert len_rgs_rec == [3, 3, 1, 3, 2, 1]
    df_ref = pdf1.iloc[:-1].append(pdf2.iloc[1:4]).append(pdf2.iloc[5:]).reset_index(drop=True)
    assert pf_rec.to_pandas().equals(df_ref)


def test_pandas_appending_duplicates_on_a_list(tmp_path):
    # Validate same as previous test, but 'duplicates_on' is a list:
    # - 'sharp starts' (meaning that bins splitting the dataframe to be written
    #   are adjusted so as not to fall in the middle of duplicates.)
    # - is also tested the index 'a' being added to 'duplicates_on', as the 2
    #   last values in 'pdf2' are not dropped despite being duplicates on 'b'.
    # - drop duplicates, keep 'last.
    # rgs                  [0, , ,1, , ,2, ]
    # idx                  [0, , ,3, , ,6, ]
    # a (ordered_on)       [0,1,2,3,3,3,4,5]
    # b (duplicates_on)    [0, , ,3, , ,6, ]
    # c (duplicate last)   [0,0,0,0,0,0,0,0]
    # a (new data)                         [5,5,5,5,6,6, 7, 8]
    # b (new data)                         [7,7,8,9,9,9,10,10]
    # c (new data)                         [1,2,3,4,5,6, 7, 8]
    # 3 duplicates (on c)                 x x x   x x x
    # rgs (new data)       [0, , ,1, , ,2,x,x,3, , ,x,4,  , 5]
    # idx                  [0, , ,3, , ,6,    7, , , 10,  ,  ]
    a1 = [0, 1, 2, 3, 3, 3, 4, 5]
    len_a1 = len(a1)
    b1 = range(len_a1)
    c1 = [0] * len_a1
    pdf1 = pDataFrame({"a": a1, "b": b1, "c": c1})
    dn = os_path.join(tmp_path, "test")
    fp_write(dn, pdf1, row_group_offsets=[0, 3, 6], file_scheme="hive", write_index=False)
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [3, 3, 2]
    max_row_group_size = 3
    a2 = [5, 5, 5, 5, 6, 6, 7, 8]
    len_a2 = len(a2)
    b2 = [7, 7, 8, 9, 9, 9, 10, 10]
    c2 = arange(len_a2) + 1
    pdf2 = pDataFrame({"a": a2, "b": b2, "c": c2})
    # 'ordered_on' with 'a', duplicates on 'b' ('a' added implicitly)
    ps_write(dn, pdf2, max_row_group_size=max_row_group_size, ordered_on="a", duplicates_on=["b"])
    pf_rec = ParquetFile(dn)
    len_rgs_rec = [rg.num_rows for rg in pf_rec.row_groups]
    assert len_rgs_rec == [3, 3, 1, 3, 2, 1]
    df_ref = pdf1.iloc[:-1].append(pdf2.iloc[1:4]).append(pdf2.iloc[5:]).reset_index(drop=True)
    assert pf_rec.to_pandas().equals(df_ref)


def test_pandas_appending_span_several_rgs(tmp_path):
    # Validate:
    # - sorting on 'ordered_on' of data which is the concatenation of existing
    #   data and new data.
    # rgs                  [0, , ,1, , ,2, ]
    # idx                  [0, , ,3, , ,6, ]
    # a                    [0,1,2,3,3,3,4,5]
    # b                    [0, , ,3, , ,6, ]
    # c                    [0,0,0,0,0,0,0,0]
    # a (new data, ordered_on)        [3,  5,5, 5,6, 6, 7, 8]
    # b (new data, duplicates_on)     [7,  7,8, 9,9, 9,10,10]
    # c (new data, check last)        [1,  2,3, 4,5, 6, 7, 8]
    # 2 duplicates (on b)                 xx,   x x  x, x  x
    # rgs (new)            [0, , ,1, ,,, ,x2, , , x  4, x, 5]
    # idx                  [0, , ,3, ,,, , 8, , ,   11,  ,13]
    a1 = [0, 1, 2, 3, 3, 3, 4, 5]
    len_a1 = len(a1)
    b1 = range(len_a1)
    c1 = [0] * len_a1
    pdf1 = pDataFrame({"a": a1, "b": b1, "c": c1})
    dn = os_path.join(tmp_path, "test")
    fp_write(dn, pdf1, row_group_offsets=[0, 3, 6], file_scheme="hive", write_index=False)
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [3, 3, 2]
    max_row_group_size = 3
    a2 = [3, 5, 5, 5, 6, 6, 7, 8]
    len_a2 = len(a2)
    b2 = [7, 7, 8, 9, 9, 9, 10, 10]
    c2 = arange(len_a2) + 1
    pdf2 = pDataFrame({"a": a2, "b": b2, "c": c2})
    # 'ordered_on' with 'a', duplicates on 'b' ('a' added implicitly)
    ps_write(dn, pdf2, max_row_group_size=max_row_group_size, ordered_on="a", duplicates_on="b")
    pf_rec = ParquetFile(dn)
    len_rgs_rec = [rg.num_rows for rg in pf_rec.row_groups]
    assert len_rgs_rec == [3, 5, 3, 2, 1]
    a_ref = [0, 1, 2, 3, 3, 3, 3, 4, 5, 5, 5, 6, 7, 8]
    b_ref = [0, 1, 2, 3, 4, 5, 7, 6, 7, 8, 9, 9, 10, 10]
    c_ref = [0, 0, 0, 0, 0, 0, 1, 0, 2, 3, 4, 6, 7, 8]
    df_ref = pDataFrame({"a": a_ref, "b": b_ref, "c": c_ref})
    assert pf_rec.to_pandas().equals(df_ref)


# inserting data in the middle of existing one: check row groups are sorted.


#    dirpath: str,
#    data: Union[pDataFrame, vDataFrame],
#    max_row_group_size: int = None,
#    compression: str = COMPRESSION,
#    cmidx_expand: bool = False,
#    cmidx_levels: List[str] = None,
#    ordered_on: Union[str, Tuple[str]] = None,
#    duplicates_on: Union[str, List[str]] = None,
#    irgs_max: int = None,


# Test coalescing of 1st row group

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
