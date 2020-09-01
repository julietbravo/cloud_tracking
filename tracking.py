import matplotlib.pyplot as pl
import netCDF4 as nc4
import xarray as xr
import numpy as np

from numba import jit


@jit(nopython=True, nogil=True, fastmath=True)
def grow_cell(cloud_id, mask, uid, i, j, itot, jtot):
    """
    Recursive function to find connected cloud (or cloud core) regions.
    """

    # Account for cyclic BCs in east/west/north/south search:
    im = i-1 if i>0 else itot-1
    ip = i+1 if i<itot-1 else 0
    jm = j-1 if j>0 else jtot-1
    jp = j+1 if j<jtot-1 else 0

    cloud_id[j,i] = uid

    # West
    if mask[j,im] and cloud_id[j,im] != uid:
        grow_cell(cloud_id, mask, uid, im, j, itot, jtot)

    # East
    if mask[j,ip] and cloud_id[j,ip] != uid:
        grow_cell(cloud_id, mask, uid, ip, j, itot, jtot)

    # South
    if mask[jm,i] and cloud_id[jm,i] != uid:
        grow_cell(cloud_id, mask, uid, i, jm, itot, jtot)

    # North
    if mask[jp,i] and cloud_id[jp,i] != uid:
        grow_cell(cloud_id, mask, uid, i, jp, itot, jtot)


@jit(nopython=True, nogil=True, fastmath=True)
def find_cells(cloud_id, mask, itot, jtot):
    """
    Find the individual cells
    """

    # Set clouds id's to 0 (not set)
    cloud_id[:,:] = 0

    uid = 20
    for j in range(jtot):
        for i in range(itot):
            if mask[j,i] and cloud_id[j,i] == 0:
                grow_cell(cloud_id, mask, uid, i, j, itot, jtot)
                uid += 1


def main():
    # Read the total (ice+water) water path, and
    # column max cloud core theta_v perturbation
    data_path = '/home/bart/meteo/models/microhh/cases/bomex/'
    nc_1 = nc4.Dataset(f'{data_path}/qlqicore_max_thv_prime.xy.nc')
    nc_2 = nc4.Dataset(f'{data_path}/qlqipath.xy.nc')

    # Info LES grid & time:
    x = nc_1.variables['x'][:]
    y = nc_1.variables['y'][:]
    time = nc_1.variables['time'][:]

    dx = x[1]-x[0]
    dy = y[1]-y[0]

    itot = x.size
    jtot = y.size

    xsize = x[-1]+dx/2
    ysize = y[-1]+dy/2

    thv = nc_1.variables['qlqicore_max_thv_prime'][:,:,:]
    wp  = nc_2.variables['qlqipath'][:,:,:]

    # Set masked grid points theta_v perturbation to small value, and remove mask
    thv[thv.mask] = -9999
    thv = np.array(thv)

    # Water path (ice+water, g kg-1) threshold for clouds:
    wp_thres = 5e-3
    # Cloud core theta_v perturbation threshold for cloud cores:
    thv_thres = 0.5

    t = -1

    # Masks for clouds and cloud cores:
    wp_mask  = wp  > wp_thres
    thv_mask = thv > thv_thres

    cloud_id = np.zeros((jtot, itot), dtype=np.uint16)
    core_id  = np.zeros((jtot, itot), dtype=np.uint16)

    find_cells(cloud_id, wp_mask[t,:,:],  itot, jtot)
    find_cells(core_id,  thv_mask[t,:,:], itot, jtot)


    pl.close('all'); pl.ion()

    pl.figure(figsize=(9,4))
    pl.subplot(121, aspect='equal')
    pl.title('Cloud', loc='left')
    pl.pcolormesh(x, y, cloud_id, cmap=pl.cm.terrain_r)
    pl.colorbar(shrink=0.7)
    pl.xlabel(r'$x$ (m)')
    pl.ylabel(r'$y$ (m)')

    pl.subplot(122, aspect='equal')
    pl.title('Cloud core', loc='left')
    pl.pcolormesh(x, y, core_id, cmap=pl.cm.terrain_r)
    pl.colorbar(shrink=0.7)
    pl.xlabel(r'$x$ (m)')
    pl.ylabel(r'$y$ (m)')

    pl.tight_layout()


if __name__ == '__main__':
    main()
