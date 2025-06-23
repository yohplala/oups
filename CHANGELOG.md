# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Calendar Versioning](https://calver.org/) with the
format YYYY.MM.patch.

**Legend**
- **Categories** indicate the type of changes (Tests, Code, Documentation, etc.).
- Each version represents a merged Pull Request.




## 2025.06

### [2025.06.3] - 2025-06-20

    PR #25: concurrent_opd_access - Prevent concurrent access to a same OPD.

    Files changed in this PR:
    .pre-commit-config.yaml
    oups/aggstream/aggstream.py
    oups/defines.py
    oups/store/filepath_utils.py
    oups/store/indexer.py
    oups/store/ordered_parquet_dataset/metadata_filename.py
    oups/store/ordered_parquet_dataset/ordered_parquet_dataset/__init__.py
    oups/store/ordered_parquet_dataset/ordered_parquet_dataset/base.py
    oups/store/ordered_parquet_dataset/ordered_parquet_dataset/read_only.py
    oups/store/ordered_parquet_dataset/write/write.py
    ... and 17 more files

**Concurrent access improvements** - [PR #25](https://github.com/yohplala/oups/pull/25)

- **Categories:** Tests, Code, Configuration, Storage, Aggregation
- **Files changed:** 27 files

### [2025.06.2] - 2025-06-09

    PR #24: iter_intersections - Polishing work.

    Files changed in this PR:
    .pre-commit-config.yaml
    oups/store/store/iter_intersections.py
    tests/test_store/test_store/test_intersections.py

**Row group iteration enhancements** - [PR #24](https://github.com/yohplala/oups/pull/24)

- **Categories:** Tests, Code, Configuration, Storage
- **Files changed:** 3 files

### [2025.06.1] - 2025-06-08

    PR #23: iter_interesections - Equiping 'Store' with 'iter_intersections()'.

    Files changed in this PR:
    .pre-commit-config.yaml
    oups/aggstream/aggstream.py
    oups/defines.py
    oups/numpy_utils.py
    oups/store/ordered_parquet_dataset/ordered_parquet_dataset.py
    oups/store/ordered_parquet_dataset/write/__init__.py
    oups/store/ordered_parquet_dataset/write/iter_merge_split_data.py
    oups/store/ordered_parquet_dataset/write/merge_split_strategies/__init__.py
    oups/store/ordered_parquet_dataset/write/merge_split_strategies/base.py
    oups/store/ordered_parquet_dataset/write/merge_split_strategies/n_rows_strategy.py
    ... and 16 more files

**Row group iteration enhancements** - [PR #23](https://github.com/yohplala/oups/pull/23)

- **Categories:** Tests, Code, Configuration, Storage, Aggregation
- **Files changed:** 26 files

## 2025.05

### [2025.05.2] - 2025-05-25

    PR #22: ordered_parquet_dataset - Refactoring.

    Files changed in this PR:
    .pre-commit-config.yaml
    oups/__init__.py
    oups/aggstream/aggstream.py
    oups/aggstream/cumsegagg.py
    oups/date_utils.py
    oups/defines.py
    oups/store/__init__.py
    oups/store/collection.py
    oups/store/defines.py
    oups/store/filepath_utils.py
    ... and 41 more files

**Ordered Parquet dataset functionality** - [PR #22](https://github.com/yohplala/oups/pull/22)

- **Categories:** Code, Configuration, Storage, Aggregation
- **Files changed:** 51 files

### [2025.05.1] - 2025-05-10

    PR #21: row_group_target_size - Fixing FutureWarnings with pandas freqstr.

    Files changed in this PR:
    .github/workflows/docs.yml
    .github/workflows/main.yml
    .pre-commit-config.yaml
    docs/source/parquetset.rst
    oups/aggstream/aggstream.py
    oups/aggstream/jcumsegagg.py
    oups/aggstream/segmentby.py
    oups/store/collection.py
    oups/store/defines.py
    oups/store/router.py
    ... and 30 more files

**Date-based row group sizing** - [PR #21](https://github.com/yohplala/oups/pull/21)

- **Categories:** Code, Documentation, Configuration, Storage, Aggregation
- **Files changed:** 40 files

## 2024.10

### [2024.10.1] - 2024-10-27

    PR #20: dual_row_group_size - A different 'row_group_target_size' can be
    used for storage of bins and snapshots results.

    Files changed in this PR:
    .vscode/settings.json
    oups/aggstream/aggstream.py
    tests/test_aggstream/test_aggstream_advanced.py
    tests/test_aggstream/test_aggstream_init.py
    tests/test_aggstream/test_aggstream_simple.py

**Dual row group target size strategy** - [PR #20](https://github.com/yohplala/oups/pull/20)

- **Categories:** Tests, Code, Aggregation
- **Files changed:** 5 files

## 2024.09

### [2024.09.1] - 2024-09-04

    PR #19: bin_snap_recording - Some optimizations.

    Files changed in this PR:
    oups/__init__.py
    oups/aggstream/aggstream.py
    oups/aggstream/cumsegagg.py
    oups/aggstream/utils.py
    oups/store/__init__.py
    oups/store/utils.py
    oups/store/writer.py
    tests/test_aggstream/test_aggstream_advanced.py
    tests/test_aggstream/test_aggstream_init.py
    tests/test_aggstream/test_aggstream_simple.py
    ... and 5 more files

**Bin snapshot recording** - [PR #19](https://github.com/yohplala/oups/pull/19)

- **Categories:** Tests, Code, Storage, Aggregation
- **Files changed:** 15 files

## 2024.07

### [2024.07.1] - 2024-07-26

    PR #18: only_metadata - Ability to store only metadata in data files.
    (when a warm-up period not producing results yet, but producing a
     'post_buffer' that needs to be recorded for instance)

    Files changed in this PR:
    oups/aggstream/aggstream.py
    oups/store/writer.py
    tests/test_aggstream/test_aggstream_simple.py

**Metadata-only operations** - [PR #18](https://github.com/yohplala/oups/pull/18)

- **Categories:** Tests, Code, Storage, Aggregation
- **Files changed:** 3 files

## 2024.06

### [2024.06.1] - 2024-06-16

    PR #17: pre_buffer - Formatting.

    Files changed in this PR:
    oups/aggstream/aggstream.py
    tests/test_aggstream/test_aggstream_advanced.py
    tests/test_aggstream/test_aggstream_init.py
    tests/test_aggstream/test_aggstream_simple.py

**Pre-buffer functionality** - [PR #17](https://github.com/yohplala/oups/pull/17)

- **Categories:** Tests, Code, Aggregation
- **Files changed:** 4 files

## 2024.04

### [2024.04.1] - 2024-04-17

    PR #16: filters - Small renaming for easier reading.

    Files changed in this PR:
    .pre-commit-config.yaml
    oups/__init__.py
    oups/aggstream/__init__.py
    oups/aggstream/aggstream.py
    oups/aggstream/cumsegagg.py
    oups/aggstream/jcumsegagg.py
    oups/aggstream/segmentby.py
    oups/aggstream/utils.py
    oups/store/collection.py
    oups/store/indexer.py
    ... and 17 more files

**Filters** - [PR #16](https://github.com/yohplala/oups/pull/16)

- **Categories:** Tests, Code, Configuration, Storage, Aggregation
- **Files changed:** 27 files

## 2023.12

### [2023.12.1] - 2023-12-27

    PR #15: aggstream - Cleaning in Store things as a package and removing 'tcut'.

    Files changed in this PR:
    .pre-commit-config.yaml
    oups/__init__.py
    oups/chainagg.py
    oups/store/__init__.py
    oups/store/collection.py
    oups/store/defines.py
    oups/store/indexer.py
    oups/store/router.py
    oups/store/utils.py
    oups/store/writer.py
    ... and 23 more files

**Aggstream** - [PR #15](https://github.com/yohplala/oups/pull/15)

- **Categories:** Tests, Code, Configuration, Storage
- **Files changed:** 33 files

## 2023.11

### [2023.11.1] - 2023-11-07

    PR #14: binary_md - Storing binary metadata.

    Files changed in this PR:
    .pre-commit-config.yaml
    oups/__init__.py
    oups/chainagg.py
    oups/collection.py
    oups/cumsegagg.py
    oups/defines.py
    oups/indexer.py
    oups/jcumsegagg.py
    oups/router.py
    oups/segmentby.py
    ... and 17 more files

**Binary metadata** - [PR #14](https://github.com/yohplala/oups/pull/14)

- **Categories:** Tests, Code, Configuration
- **Files changed:** 27 files

## 2023.05

### [2023.05.1] - 2023-05-21

    PR #13: cumsegagg_restart - Test for 'cumesgagg()' restart feature.

    Files changed in this PR:
    .pre-commit-config.yaml
    oups/cumsegagg.py
    oups/indexer.py
    oups/jcumsegagg.py
    oups/segmentby.py
    pyproject.toml
    tests/test_cumsegagg.py
    tests/test_cumsegagg_restart.py
    tests/test_jcumsegagg.py
    tests/test_segmentby.py
    ... and 1 more files

**Cumsegagg restart** - [PR #13](https://github.com/yohplala/oups/pull/13)

- **Categories:** Tests, Code, Configuration
- **Files changed:** 11 files

## 2023.03

### [2023.03.1] - 2023-03-23

    PR #12: snapshot - Minor optimizations.

    Files changed in this PR:
    .github/workflows/docs.yml
    .github/workflows/main.yml
    .pre-commit-config.yaml
    oups/chainagg.py
    oups/cumsegagg.py
    oups/jcumsegagg.py
    oups/segmentby.py
    pyproject.toml
    tests/test_cumsegagg.py
    tests/test_jcumsegagg.py
    ... and 2 more files

**Snapshot** - [PR #12](https://github.com/yohplala/oups/pull/12)

- **Categories:** Tests, Code, Configuration
- **Files changed:** 12 files

## 2022.09

### [2022.09.2] - 2022-09-19

    PR #11: aggstream - Renaming 'streamagg' to 'chainagg'.

    Files changed in this PR:
    .pre-commit-config.yaml
    oups/__init__.py
    oups/chainagg.py
    oups/collection.py
    oups/utils.py
    pyproject.toml
    tests/test_chainagg_multi.py
    tests/test_chainagg_simple.py

**Aggstream** - [PR #11](https://github.com/yohplala/oups/pull/11)

- **Categories:** Tests, Code, Configuration
- **Files changed:** 8 files

### [2022.09.1] - 2022-09-12

    PR #10: aggstream - Minor cleanings.

    Files changed in this PR:
    oups/collection.py
    oups/streamagg.py
    oups/utils.py
    oups/writer.py
    tests/test_collection.py
    tests/test_metadata.py
    tests/test_streamagg.py
    tests/test_streamagg2.py
    tests/test_utils.py

**Aggstream** - [PR #10](https://github.com/yohplala/oups/pull/10)

- **Categories:** Tests, Code
- **Files changed:** 9 files

## 2022.08

### [2022.08.1] - 2022-08-11

    PR #8: cum_agg - Cleanings in 'aggstream'.

    Files changed in this PR:
    .pre-commit-config.yaml
    oups/__init__.py
    oups/indexer.py
    oups/router.py
    oups/streamagg.py
    oups/writer.py
    pyproject.toml
    tests/test_metadata.py
    tests/test_router.py
    tests/test_streamagg.py
    ... and 1 more files

**Aggstream** - [PR #8](https://github.com/yohplala/oups/pull/8)

- **Categories:** Tests, Code, Configuration
- **Files changed:** 11 files

## 2022.01

### [2022.01.4] - 2022-01-17

    PR #6: update - fastparquet version update.

    Files changed in this PR:
    .pre-commit-config.yaml
    oups/router.py
    oups/utils.py
    oups/writer.py
    pyproject.toml
    tests/test_collection.py
    tests/test_utils.py
    tests/test_writer.py

**Dependency updates** - [PR #6](https://github.com/yohplala/oups/pull/6)

- **Categories:** Tests, Code, Configuration
- **Files changed:** 8 files

### [2022.01.3] - 2022-01-07

    PR #5: doc - Adding API reference.

    Files changed in this PR:
    docs/source/api.rst
    docs/source/conf.py
    docs/source/index.rst

**Documentation** - [PR #5](https://github.com/yohplala/oups/pull/5)

- **Categories:** Code, Documentation
- **Files changed:** 3 files

### [2022.01.2] - 2022-01-06

    PR #4: CI - Some simplifications.

    Files changed in this PR:
    .github/workflows/docs.yml
    .github/workflows/main.yml
    .pre-commit-config.yaml
    pyproject.toml

**CI** - [PR #4](https://github.com/yohplala/oups/pull/4)

- **Categories:** Code, Configuration
- **Files changed:** 4 files

### [2022.01.1] - 2022-01-05

    PR #3: sphinx - Some adjustments.

    Files changed in this PR:
    .github/workflows/docs.yml
    .github/workflows/main.yml
    README.md
    README.rst
    docs/Makefile
    docs/make.bat
    docs/source/conf.py
    docs/source/index.rst
    docs/source/indexing.rst
    docs/source/install.rst
    ... and 1 more files

**Documentation** - [PR #3](https://github.com/yohplala/oups/pull/3)

- **Categories:** Code, Documentation
- **Files changed:** 11 files

## 2021.12

### [2021.12.2] - 2021-12-28

    PR #2: Store.

    Files changed in this PR:
    .github/workflows/main.yml
    .pre-commit-config.yaml
    oups/__init__.py
    oups/collection.py
    oups/defines.py
    oups/indexer.py
    oups/router.py
    oups/utils.py
    oups/writer.py
    pyproject.toml
    ... and 9 more files

**Store** - [PR #2](https://github.com/yohplala/oups/pull/2)

- **Categories:** Tests, Code, Configuration
- **Files changed:** 19 files

### [2021.12.1] - 2021-12-28

    PR #1: indexer - testing.

    Files changed in this PR:
    oups/__init__.py
    oups/defines.py
    oups/indexer.py
    setup.py
    tests/test_indexer.py

**Indexer** - [PR #1](https://github.com/yohplala/oups/pull/1)

- **Categories:** Tests, Code, Configuration
- **Files changed:** 5 files
