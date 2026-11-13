"""Corrige WholeExtent/Origin das saidas gravadas antes do ajuste (offset em y vai para Origin)."""
import sys, h5py, numpy as np
for fn in sys.argv[1:]:
    with h5py.File(fn, "r+") as f:
        g = f["VTKHDF"]
        e = g.attrs["WholeExtent"]
        if e[2] != 0:
            g.attrs["Origin"] = np.array([0, e[2], 0], dtype=np.float64)
            g.attrs["WholeExtent"] = np.array([0, e[1], 0, e[3] - e[2], 0, e[5]], dtype=np.int64)
