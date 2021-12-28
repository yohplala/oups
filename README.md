# OUPS
Ordered Updatable Parquet Store

Help organize parquet datasets with objective to deliver convenience function for updating ordered datasets.

```python
from os import path as os_path
import pandas as pd
from oups import toplevel, ParquetSet

# Define the schema to generate directory path where will be stored the data.
@toplevel
class DataSetIndex:
    country: str
    city: str

# Initialize a parquet store.
store_path = os_path.expanduser('~/Documents/code/data/parquet_store')
ps = ParquetSet(store_path, DataSetIndex)

# Start growing the store with one dataset.
idx1 = DataSetIndex('germany','berlin')
df = pd.DataFrame({'timestamp':pd.date_range('2021/01/01', '2021/01/05', freq='1D'),
                   'temperature':range(10,15)})
ps[idx1] = df

# Get the data back.
ps[idx1].pdf
Out[1]: 
   timestamp  temperature
0 2021-01-01           10
1 2021-01-02           11
2 2021-01-03           12
3 2021-01-04           13
4 2021-01-05           14

# Add new datasets as needed.
idx2 = DataSetIndex('japan','tokyo')
df = pd.DataFrame({'timestamp':pd.date_range('2020/01/01', '2020/01/05', freq='1D'),
                   'temperature':range(15,20)})
ps[idx2] = df
```

Can query
  - fastparquet parquet file `ps[idx].pf`,
  - pandas dataframe `ps[idx].pdf`,
  - or vaex dataframe `ps[idx].vdf`.

Currently append to existing dataset.
```python
ps[idx1] = df

ps[idx1].pdf
Out[2]: 
   timestamp  temperature
0 2021-01-01           10
1 2021-01-02           11
2 2021-01-03           12
3 2021-01-04           13
4 2021-01-05           14
5 2021-01-01           10
6 2021-01-02           11
7 2021-01-03           12
8 2021-01-04           13
9 2021-01-05           14
```

Function to update ordered dataset, with management of duplicates is a target.

Goodies
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
```
