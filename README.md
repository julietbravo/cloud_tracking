# 2D cloud tracking algorithm

2D cloud tracking algorithm in Python/Numba, based on:

Heus, T., & Seifert, A. (2013). _Automated tracking of shallow cumulus clouds in large domain, long duration large eddy simulations_. Geoscientific Model Development, 6(4), 1261.

For running the cloud tracking on MicroHH output, the following cross-sections are required:

`crosslist = qlqipath,qlqibase,qlqitop,qlqicore_max_thv_prime`

Two versions are currently available:
- `cloud_tracking.py` uses the same recursive method as Heus & Seifert, which Python does not like. It requires setting the limit on recursive calls very high (especially when using time tracking), which apparently does not work on Windows and Mac OS.
- `cloud_tracking_v2.py` uses a non-recursive method, which should work on all systems. The results are perfectly identical to the output from `cloud_tracking.py`.
