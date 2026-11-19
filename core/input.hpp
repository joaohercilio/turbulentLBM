#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <hdf5.h>

#include "definitions.hpp"


inline void readPointData(hid_t file, const char* name, dfloat* data, const int NX, const int NY, const int NZ)
{
    const std::string path = std::string("VTKHDF/PointData/") + name;
    hid_t dset = H5Dopen(file, path.c_str(), H5P_DEFAULT);
    if (dset < 0)
    {
        std::printf("dataset %s nao encontrado\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }

    hid_t space = H5Dget_space(dset);
    hsize_t dims[3] = {0, 0, 0};
    const int rank = H5Sget_simple_extent_ndims(space);
    H5Sget_simple_extent_dims(space, dims, nullptr);
    H5Sclose(space);
    if (rank != 3 || dims[0] != hsize_t(NZ) || dims[1] != hsize_t(NY) || dims[2] != hsize_t(NX))
    {
        std::printf("%s tem shape diferente de [NZ, NY, NX] = [%d, %d, %d]\n", path.c_str(), NZ, NY, NX);
        std::exit(EXIT_FAILURE);
    }

    const hid_t type = std::is_same_v<dfloat, float> ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;
    H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dset);
}
