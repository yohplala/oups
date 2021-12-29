# Welcome to OUPS!

## 1. What is OUPS?
OUPS stands for Ordered Updatable Parquet Store.

OUPS aims primarily at:
- helping in the organization of parquet datasets,
- delivering convenience function for updating ordered datasets.

### 1.a Organized datasets.

#### `@toplevel` and `@sublevel` class decorators.
In OUPS, datasets are organized in separate folders, which naming follows a user-defined schema.
This schema is formalized by use of `@toplevel` (and optionally one or several `@sublevel`) class decorator(s).
A class decorated with `@toplevel` defines necessarily the first directory level (while `@sublevel` has to be used for sub-directories).
 These decorators act in the same way than `@dataclass`. They actually wrap it and force some settings.

```python
from oups import toplevel

# Define the schema to generate directory path where will be stored the data.
@toplevel
class DatasetIndex:
    country: str
    city: str
```

#### `ParquetSet` instance.
This schema (decorated class) is then used when instantiating a `ParquetSet`, in other words, a collection of parquet datasets.

```python
from os import path as os_path
from oups import ParquetSet

# Initialize a parquet store, specifying:
# - where datasets have to be recorded (or read from);
# - the schema to be used for deriving path to individual dataset.
store_path = os_path.expanduser('~/Documents/code/data/parquet_store')
ps = ParquetSet(store_path, DatasetIndex)
```

#### OUPS in action.
All is now set to create a new dataset, from new data.
```python
import pandas as pd
# Key to a first dataset, for some temperature records related to Berlin.
idx1 = DatasetIndex('germany','berlin')
df1 = pd.DataFrame({'timestamp':pd.date_range('2021/01/01', '2021/01/05', freq='1D'),
                   'temperature':range(10,15)})
# Initiate the new parquet dataset.
ps[idx1] = df1
```

### 1.b Updating ordered datasets.
No function yet has been implemented.
It is however a target to deliver it.

## 2. Requirements
- python (3.7 or higher)
- pandas (1.3.4 or higher)
- vaex (4.6.0 or higher)
- fastparquet, specific branch (PR pending)
```bash
git+https://github.com/yohplala/fastparquet@cmidx_write_rg
```
- sortedcontainers

## 3. Why OUPS?
As a self-taught data wrangler, I have been in need of a solution to organize a collection of ordered datasets, more specifically time series. Hence a first step has been the implementation of `@toplevel` and `@sublevel` class decorators.

Other libraries out there already exist to manage collections of datasets,
- many that I have not tested, for instance [Arctic](https://github.com/man-group/arctic)
- one that I have tested, [pystore](https://github.com/ranaroussi/pystore). Being based on Dask, it supports parallelized reading/writing out of the box. Its update logic can be reviewed in [`collection.py`](https://github.com/ranaroussi/pystore/blob/ed9beca774312811527c80d199c3cf437623477b/pystore/collection.py#L181). Not elaborating about [possible performance issues](https://github.com/ranaroussi/pystore/issues/56), and only focusing on usability, current procedure implies that any duplicate rows, considering all columns, but not the index (which is necessarily a `Datetimeindex` as per `pystore` implementation), be dropped, except last (discussed in [ticket #43](https://github.com/ranaroussi/pystore/issues/43)). But this hard-coded logic may not suit all dataflows.

In comparison, current version of OUPS:
- is not based on Dask but directly on [fastparquet](https://fastparquet.readthedocs.io/en/latest/). No parallelized reading/writing is yet possible.
- only appends new data, without dropping duplicates. It is however a target to propose an update function with a user-defined logic for dropping duplicates.

## 4. Usage notes.

### 4.a Dataframe format.
- OUPS accepts [pandas](https://github.com/pandas-dev/pandas) or [vaex](https://github.com/vaexio/vaex) dataframes.
- Row index is dropped when recording. If the index of your dataframe is meaningful, make sure to reset it as a column.
```python
pandas_df = pandas_df.reset_index()
```
This only applies for pandas dataframes, as vaex's ones have row index.
- Column multi-index can be recorded. Here again vaex has no support for column multi-index. But if your vaex dataframe comes from a pandas one, with column multi-index, you can expand it again at recording.
```python
# with vaex_df created from a pandas df with column multi-index.
ps[idx] = {'cmidx_expand'=True}, vaex_df
```

### 4.b Overview of OUPS features.

#### Get your data back.
OUPS returns data either through 'handles' (vaex dataframe or fastparquet parquet file) or directly as a pandas dataframe.
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
Currently OUPS only append new data to existing one, with no additional processing (no drop of duplicate, no re-ordering of data when 'old' data is being added).
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
# Schema definition needs to be available.
ps2 = ParquetSet(store_path, DatasetIndex)
ps2
Out[6]: japan-tokyo
```
