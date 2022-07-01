import matplotlib.pyplot as pl
import netCDF4 as nc4
import xarray as xr
import numpy as np

from numba import jit, prange

import resource
import sys
resource.setrlimit(resource.RLIMIT_STACK, [100000000000, resource.RLIM_INFINITY])
sys.setrecursionlimit(1000000000)


@jit(nopython=True, nogil=True, fastmath=True)
def no_overlap(base1, top1, base2, top2):
    """
    Check for possible cloud overlap, where the cloud top of one
    grid point is below the cloud base of another
    """

    if base1 <= top2 and top1 >= base2:
        return True
    else:
        return False


@jit(nopython=True, nogil=True, fastmath=True)
def grow_cell(
        cloud_id, mask, uid, base, top,
        i, j, t, nt, itot, jtot, time_tracking):
    """
    Recursive function to find connected cloud (or cloud core) regions.
    """

    # Account for cyclic BCs in east/west/north/south search:
    im = i-1 if i>0 else itot-1
    ip = i+1 if i<itot-1 else 0
    jm = j-1 if j>0 else jtot-1
    jp = j+1 if j<jtot-1 else 0

    cloud_id[t,j,i] = uid

    # West
    if mask[t,j,im] and cloud_id[t,j,im] == 0:
        if no_overlap(base[t,j,i], top[t,j,i], base[t,j,im], top[t,j,im]):
            grow_cell(cloud_id, mask, uid, base, top, im, j, t, nt, itot, jtot, time_tracking)

    # East
    if mask[t,j,ip] and cloud_id[t,j,ip] == 0:
        if no_overlap(base[t,j,i], top[t,j,i], base[t,j,ip], top[t,j,ip]):
            grow_cell(cloud_id, mask, uid, base, top, ip, j, t, nt, itot, jtot, time_tracking)

    # South
    if mask[t,jm,i] and cloud_id[t,jm,i] == 0:
        if no_overlap(base[t,j,i], top[t,j,i], base[t,jm,i], top[t,jm,i]):
            grow_cell(cloud_id, mask, uid, base, top, i, jm, t, nt, itot, jtot, time_tracking)

    # North
    if mask[t,jp,i] and cloud_id[t,jp,i] == 0:
        if no_overlap(base[t,j,i], top[t,j,i], base[t,jp,i], top[t,jp,i]):
            grow_cell(cloud_id, mask, uid, base, top, i, jp, t, nt, itot, jtot, time_tracking)

    if time_tracking:
        # Backward in time:
        if t > 0 and mask[t-1,j,i] and cloud_id[t-1,j,i] == 0:
            if no_overlap(base[t,j,i], top[t,j,i], base[t-1,j,i], top[t-1,j,i]):
                grow_cell(cloud_id, mask, uid, base, top, i, j, t-1, nt, itot, jtot, time_tracking)

        # Forward in time:
        if t < nt-1 and mask[t+1,j,i] and cloud_id[t+1,j,i] == 0:
            if no_overlap(base[t,j,i], top[t,j,i], base[t+1,j,i], top[t+1,j,i]):
                grow_cell(cloud_id, mask, uid, base, top, i, j, t+1, nt, itot, jtot, time_tracking)


@jit(nopython=False, nogil=True, fastmath=True)
def find_cells(cloud_id, mask, base, top, nt, itot, jtot, time_tracking):
    """
    Find the individual cells (incl. time tracking)
    """

    # Cloud ID 0 = not yet assigned
    cloud_id[:,:,:] = 0

    uid = 1
    for t in range(nt):
        if not time_tracking:
            uid = 1
        for j in range(jtot):
            for i in range(itot):
                if mask[t,j,i] and cloud_id[t,j,i] == 0:
                    grow_cell(
                            cloud_id, mask, uid, base, top,
                            i, j, t, nt, itot, jtot, time_tracking)
                    uid += 1


if __name__ == '__main__':

    # Time indices to process:
    dt = 60
    t0 = 0
    t1 = 721
    time_tracking = False

    # Read the total (ice+water) water path, and
    # column max cloud core theta_v perturbation
    data_path = '.'

    nc_1 = nc4.Dataset(f'{data_path}/qlqicore_max_thv_prime.xy.nc')
    nc_2 = nc4.Dataset(f'{data_path}/qlqipath.xy.nc')
    nc_3 = nc4.Dataset(f'{data_path}/qlqibase.xy.nc')
    nc_4 = nc4.Dataset(f'{data_path}/qlqitop.xy.nc')

    # Info LES grid & time:
    x = nc_1.variables['x'][:]
    y = nc_1.variables['y'][:]
    time = nc_1.variables['time'][t0:t1]

    itot = x.size
    jtot = y.size
    nt = time.size

    # Read all data to memory (..)
    dtype = np.float32
    thv  = nc_1.variables['qlqicore_max_thv_prime'][t0:t1,:,:].astype(dtype)
    wp   = nc_2.variables['qlqipath'][t0:t1,:,:].astype(dtype)
    base = nc_3.variables['qlqibase'][t0:t1,:,:].astype(dtype)
    top  = nc_4.variables['qlqitop'] [t0:t1,:,:].astype(dtype)

    # Set masked grid points theta_v perturbation to small value, and remove mask
    thv[thv.mask] = -10
    thv = np.array(thv, dtype=dtype)

    # Water path (ice+water, g kg-1) threshold for clouds:
    wp_thres = 5e-3
    # Cloud core theta_v perturbation threshold for cloud cores:
    thv_thres = 0.5

    # Masks for clouds and cloud cores:
    wp_mask  = wp  > wp_thres
    thv_mask = thv > thv_thres

    # Cleanup!
    del thv
    del wp

    # Do the cloud tracking
    dtype_int = np.uint32
    cloud_id = np.zeros((nt, jtot, itot), dtype=dtype_int)
    core_id  = np.zeros((nt, jtot, itot), dtype=dtype_int)

    print('Tracking clouds')
    find_cells(cloud_id, wp_mask,  base, top, nt, itot, jtot, time_tracking)
    print('Tracking cores')
    find_cells(core_id,  thv_mask, base, top, nt, itot, jtot, time_tracking)

    # Save ID's in NetCDF format
    if time_tracking:
        nc_names = ['cloud_tracking_time.nc', 'core_tracking_time.nc']
    else:
        nc_names = ['cloud_tracking_inst.nc', 'core_tracking_inst.nc']
    values   = [cloud_id, core_id]

    for nc_name, ids in zip(nc_names, values):
        nc = nc4.Dataset(nc_name, 'w')

        dim_x = nc.createDimension('x', itot)
        dim_y = nc.createDimension('y', jtot)
        dim_t = nc.createDimension('time', nt)

        var_x = nc.createVariable('x', np.float32, 'x')
        var_y = nc.createVariable('y', np.float32, 'y')
        var_t = nc.createVariable('time', np.float32, 'time')

        var_id = nc.createVariable('id', dtype_int, ('time','y','x'))

        var_x [:] = x
        var_y [:] = y
        var_t [:] = time
        var_id[:] = ids

        nc.close()
