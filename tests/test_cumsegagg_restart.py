#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh
"""


# from pandas.testing import assert_frame_equal
# tmp_path = os_path.expanduser('~/Documents/code/data/oups')


# Questions:
#  - in cumsegagg, when restarting with 3 empty chunks, assuming the
#    there are 2 'snaps', and the 3rd is a 'bin' (which was in progress at
#    prev iter.)
#    With proposed methodo, it is not enough to simply let the 'in-progress data'
#    from previous iteration. the new intermediate chunk needs to be created.
#
#  We should really just restart jcsagg directly, without removing empyt bins
#  If values are the same, then they are the same and that's it.
# 'pinnu' should be forwarded with its last value / should work.
# length will be detected 0
# Make test
#   - with empty snap at prev iter, then new empty snaps at next iter then end of bin.
#   - with non empty snap at prev iter, then new empty one (res forwarded?) at next iter then end of bin
# replay also test case segmentby() 5
# 5/ 'bin_by' using 'by_x_rows', 'snap_by' as Series.
#    In this test case, 'unknown_bin_end' is False after 1st data
#    chunk.
#    Last rows in 'data' are not accounted for in 1st series of
#    snapshots (8:20 snasphot not present), but is in 2nd series.
