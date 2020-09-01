import matplotlib.pyplot as pl
import netCDF4 as nc4
import xarray as xr
import numpy as np

from numba import jit

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
def grow_cell_inst(cloud_id, mask, base, top, uid, i, j, itot, jtot):
    """
    Recursive function to find connected cloud (or cloud core) regions.
    Uses one single time step, so no time tracking is included.
    """

    # Account for cyclic BCs in east/west/north/south search:
    im = i-1 if i>0 else itot-1
    ip = i+1 if i<itot-1 else 0
    jm = j-1 if j>0 else jtot-1
    jp = j+1 if j<jtot-1 else 0

    cloud_id[j,i] = uid

    # West
    if mask[j,im] and cloud_id[j,im] != uid:
        if no_overlap(base[j,i], top[j,i], base[j,im], top[j,im]):
            grow_cell_inst(cloud_id, mask, base, top, uid, im, j, itot, jtot)

    # East
    if mask[j,ip] and cloud_id[j,ip] != uid:
        if no_overlap(base[j,i], top[j,i], base[j,ip], top[j,ip]):
            grow_cell_inst(cloud_id, mask, base, top, uid, ip, j, itot, jtot)

    # South
    if mask[jm,i] and cloud_id[jm,i] != uid:
        if no_overlap(base[j,i], top[j,i], base[jm,i], top[jm,i]):
            grow_cell_inst(cloud_id, mask, base, top, uid, i, jm, itot, jtot)

    # North
    if mask[jp,i] and cloud_id[jp,i] != uid:
        if no_overlap(base[j,i], top[j,i], base[jp,i], top[jp,i]):
            grow_cell_inst(cloud_id, mask, base, top, uid, i, jp, itot, jtot)


@jit(nopython=True, nogil=True, fastmath=True)
def find_cells_inst(cloud_id, mask, base, top, itot, jtot):
    """
    Find the individual cells (single time step)
    """

    # Set clouds id's to 0 (not set)
    cloud_id[:,:] = 0

    uid = 20
    for j in range(jtot):
        for i in range(itot):
            if mask[j,i] and cloud_id[j,i] == 0:
                grow_cell_inst(cloud_id, mask, base, top, uid, i, j, itot, jtot)
                uid += 1


@jit(nopython=True, nogil=True, fastmath=True)
def grow_cell_time(cloud_id, mask, uid, base, top, i, j, t, tstart, tend, itot, jtot):
    """
    Recursive function to find connected cloud (or cloud core) regions.
    """

    # Account for cyclic BCs in east/west/north/south search:
    im = i-1 if i>0 else itot-1
    ip = i+1 if i<itot-1 else 0
    jm = j-1 if j>0 else jtot-1
    jp = j+1 if j<jtot-1 else 0

    # Offset time index in the output `cloud_id` field
    tt = t-tstart

    cloud_id[tt,j,i] = uid

    # West
    if mask[t,j,im] and cloud_id[tt,j,im] != uid:
        if no_overlap(base[t,j,i], top[t,j,i], base[t,j,im], top[t,j,im]):
            grow_cell_time(cloud_id, mask, uid, base, top, im, j, t, tstart, tend, itot, jtot)

    # East
    if mask[t,j,ip] and cloud_id[tt,j,ip] != uid:
        if no_overlap(base[t,j,i], top[t,j,i], base[t,j,ip], top[t,j,ip]):
            grow_cell_time(cloud_id, mask, uid, base, top, ip, j, t, tstart, tend, itot, jtot)

    # South
    if mask[t,jm,i] and cloud_id[tt,jm,i] != uid:
        if no_overlap(base[t,j,i], top[t,j,i], base[t,jm,i], top[t,jm,i]):
            grow_cell_time(cloud_id, mask, uid, base, top, i, jm, t, tstart, tend, itot, jtot)

    # North
    if mask[t,jp,i] and cloud_id[tt,jp,i] != uid:
        if no_overlap(base[t,j,i], top[t,j,i], base[t,jp,i], top[t,jp,i]):
            grow_cell_time(cloud_id, mask, uid, base, top, i, jp, t, tstart, tend, itot, jtot)

    # Backward in time:
    if t > tstart and mask[t-1,j,i] and cloud_id[tt-1,j,i] != uid:
        if no_overlap(base[t,j,i], top[t,j,i], base[t-1,j,i], top[t-1,j,i]):
            grow_cell_time(cloud_id, mask, uid, base, top, i, j, t-1, tstart, tend, itot, jtot)

    # Forward in time:
    if t < tend-1 and mask[t+1,j,i] and cloud_id[tt+1,j,i] != uid:
        if no_overlap(base[t,j,i], top[t,j,i], base[t+1,j,i], top[t+1,j,i]):
            grow_cell_time(cloud_id, mask, uid, base, top, i, j, t+1, tstart, tend, itot, jtot)


@jit(nopython=True, nogil=True, fastmath=True)
def find_cells_time(cloud_id, mask, base, top, tstart, tend, itot, jtot):
    """
    Find the individual cells (incl. time tracking)
    """

    # Set clouds id's to 0 (not set)
    cloud_id[:,:,:] = 0

    uid = 10
    for t in range(tstart, tend):
        tt = t-tstart
        for j in range(jtot):
            for i in range(itot):
                if mask[t,j,i] and cloud_id[tt,j,i] == 0:
                    grow_cell_time(cloud_id, mask, uid, base, top, i, j, t, tstart, tend, itot, jtot)
                    uid += 1



if __name__ == '__main__':

    # Read the total (ice+water) water path, and
    # column max cloud core theta_v perturbation
    data_path = '/home/bart/meteo/models/microhh/cases/bomex/'
    nc_1 = nc4.Dataset(f'{data_path}/qlqicore_max_thv_prime.xy.nc')
    nc_2 = nc4.Dataset(f'{data_path}/qlqipath.xy.nc')
    nc_3 = nc4.Dataset(f'{data_path}/qlqibase.xy.nc')
    nc_4 = nc4.Dataset(f'{data_path}/qlqitop.xy.nc')

    # Info LES grid & time:
    x = nc_1.variables['x'][:]
    y = nc_1.variables['y'][:]

    time = nc_1.variables['time'][:]
    nt = time.size

    dx = x[1]-x[0]
    dy = y[1]-y[0]

    itot = x.size
    jtot = y.size

    xsize = x[-1]+dx/2
    ysize = y[-1]+dy/2

    thv  = nc_1.variables['qlqicore_max_thv_prime'][:,:,:]
    wp   = nc_2.variables['qlqipath'][:,:,:]
    base = nc_3.variables['qlqibase'][:,:,:]
    top  = nc_4.variables['qlqitop'][:,:,:]

    # Set masked grid points theta_v perturbation to small value, and remove mask
    thv[thv.mask] = -9999
    thv = np.array(thv)

    # Water path (ice+water, g kg-1) threshold for clouds:
    wp_thres = 5e-3
    # Cloud core theta_v perturbation threshold for cloud cores:
    thv_thres = 0.5

    # Masks for clouds and cloud cores:
    wp_mask  = wp  > wp_thres
    thv_mask = thv > thv_thres

    #
    # Tracking incl. time change
    #
    if True:
        t0 = 120
        t1 = 361
        nt = t1-t0

        cloud_id = np.zeros((nt, jtot, itot), dtype=np.uint16)
        core_id  = np.zeros((nt, jtot, itot), dtype=np.uint16)

        find_cells_time(cloud_id, wp_mask,  base, top, t0, t1, itot, jtot)
        find_cells_time(core_id,  thv_mask, base, top, t0, t1, itot, jtot)

        uid_max = cloud_id.max()

        for t in range(t0, t1, 1):
            tt = t-t0

            pl.close('all'); pl.ioff()

            pl.figure(figsize=(5,4))
            pl.subplot(111, aspect='equal')
            pl.title('Cloud, t={} s'.format(time[t]), loc='left')
            pl.imshow(
                    cloud_id[tt,:,:], origin='lower', extent=[0,xsize,0,ysize],
                    interpolation='nearest', cmap=pl.cm.terrain_r,
                    vmin=0, vmax=uid_max)
            pl.colorbar(shrink=0.7)
            pl.xlabel(r'$x$ (m)')
            pl.ylabel(r'$y$ (m)')

            pl.tight_layout()
            pl.savefig('figs/t{0:04d}.png'.format(t))


    #
    # "Tracking" for a single time step
    #
    if False:
        cloud_id = np.zeros((jtot, itot), dtype=np.uint16)
        core_id  = np.zeros((jtot, itot), dtype=np.uint16)

        t = -1
        find_cells_inst(cloud_id, wp_mask[t,:,:],  base[t,:,:], top[t,:,:], itot, jtot)
        find_cells_inst(core_id,  thv_mask[t,:,:], base[t,:,:], top[t,:,:], itot, jtot)

        pl.close('all'); pl.ion()

        pl.figure(figsize=(9,4))
        pl.subplot(121, aspect='equal')
        pl.title('Cloud', loc='left')
        pl.imshow(
                cloud_id, origin='lower', extent=[0,xsize,0,ysize],
                interpolation='nearest', cmap=pl.cm.terrain_r)
        pl.colorbar(shrink=0.7)
        pl.xlabel(r'$x$ (m)')
        pl.ylabel(r'$y$ (m)')

        pl.subplot(122, aspect='equal')
        pl.title('Cloud core', loc='left')
        pl.imshow(
                core_id, origin='lower', extent=[0,xsize,0,ysize],
                interpolation='nearest', cmap=pl.cm.terrain_r)
        pl.colorbar(shrink=0.7)
        pl.xlabel(r'$x$ (m)')
        pl.ylabel(r'$y$ (m)')

        pl.tight_layout()
