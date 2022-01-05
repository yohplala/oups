# Welcome to OUPS!

## What is OUPS?
*oups* stands for Ordered Updatable Parquet Store.

*oups* Python library provides convenience functions and class to manage a collection of parquet datasets. This includes mostly collection indexing, and ability to update ordered datasets.

Please, head to the [documentation](https://yohplala.github.io/oups/) to get acquainted!


# WiP

Below sections are gradually integrated into the documentation. Just keeping them here at the moment.

## 3. Why OUPS?
As a self-taught data wrangler, I have been confronted with managing 'large-size' collections of ordered datasets, more specifically time series.
These datasets may contain different data (from different channels or feeds) or be the results of different processings of the same raw data.
To ease their management, i.e. organize how storing these datasets, a first step in `oups` has then been the implementation of `@toplevel` and `@sublevel` class decorators.

Other libraries out there already exist to manage collections of datasets,
- many that I have not tested, for instance [Arctic](https://github.com/man-group/arctic)
- one that I have tested, [pystore](https://github.com/ranaroussi/pystore). Being based on Dask, it supports parallelized reading/writing out of the box. Its update logic can be reviewed in [`collection.py`](https://github.com/ranaroussi/pystore/blob/ed9beca774312811527c80d199c3cf437623477b/pystore/collection.py#L181). Not elaborating about its [possible performance issues](https://github.com/ranaroussi/pystore/issues/56), and only focusing on this logic applicability, current procedure implies that any duplicate rows, considering all columns, but not the index (which is necessarily a `Datetimeindex` as per `pystore` implementation), be dropped, except last. But this hard-coded logic [may not suit all dataflows](https://github.com/ranaroussi/pystore/issues/43).

In comparison, current version of `oups`,
- is not based on Dask but directly on [fastparquet](https://fastparquet.readthedocs.io/en/latest/). No parallelized reading/writing is yet possible.
- only appends new data, without dropping duplicates. It is however a target to propose an `update` function with a user-defined logic for dropping duplicates.

## 4. Usage notes.

### 4.a Dataframe format.
- `oups` accepts [pandas](https://github.com/pandas-dev/pandas) or [vaex](https://github.com/vaexio/vaex) dataframes.
- Row index is dropped when recording. If the index of your dataframe is meaningful, make sure to reset it as a column.
```python
pandas_df = pandas_df.reset_index()
```
This only applies for pandas dataframes, as vaex ones have no row index.
- Column multi-index can be recorded. Here again vaex has no support for column multi-index. But if your vaex dataframe comes from a pandas one initially with column multi-index, you can expand it again at recording.
```python
# With 'vaex_df' created from a pandas dataframe with column multi-index.
ps[idx] = {'cmidx_expand'=True}, vaex_df
```

### 4.b Overview of OUPS features.

#### Get your data back.
`oups` returns data either through 'handles' (vaex dataframe or fastparquet parquet file) or directly as a pandas dataframe.
  - fastparquet parquet file `ps[idx].pf`,
  - vaex dataframe `ps[idx].vdf`.
  - or pandas dataframe `ps[idx].pdf`,
```python
# Initial example continued.
ps[idx1].pdf
Out[1]:
   timestamp  temperature
0 2021-01-01           10
1 2021-01-02           11
2 2021-01-03           12
3 2021-01-04           13
4 2021-01-05           14
```

#### Add new datasets.
To record a new dataset into an existing collection, the same `@toplevel`-decorated class has to be instanced anew with different values to create a new 'index'.
```python
idx2 = DatasetIndex('japan','tokyo')
df2 = pd.DataFrame({'timestamp':pd.date_range('2020/01/01', '2020/01/05', freq='1D'),
                    'temperature':range(15,20)})
ps[idx2] = df2
```

#### Update existing datasets.
Currently `oups` only append new data to existing one, with no additional processing (in particular, no dropping of duplicates, nor re-ordering of data when 'old' data is being added).
```python
ps[idx1] = df1

ps[idx1].pdf
Out[2]:
   timestamp  temperature
0 2021-01-01           10
1 2021-01-02           11
2 2021-01-03           12
3 2021-01-04           13
4 2021-01-05           14
5 2021-01-01           10    # new appended data
6 2021-01-02           11
7 2021-01-03           12
8 2021-01-04           13
9 2021-01-05           14
```

#### Other "goodies".
```python
# Review store content.
ps
Out[3]:
germany-berlin
japan-tokyo

# Get number of datasets.
len(ps)
Out[4]: 2

# Delete a dataset (delete data from disk).
del ps[idx1]
ps
Out[5]: japan-tokyo

# 'Discover' an existing dataset collection.
# Initial schema definition is needed.
ps2 = ParquetSet(store_path, DatasetIndex)
ps2
Out[6]: japan-tokyo
```
