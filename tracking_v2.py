import netCDF4 as nc4
import xarray as xr
import numpy as np
from numba import jit

# Increase this value if the queue indexing goes out-of-bounds...
QUEUE_SIZE = 1000000

@jit(nopython=True, nogil=True, fastmath=True)
def cyclic_lims(n, ntot):
    """
    Apply periodic BCs to indexing.
    """
    if n == -1:
        return ntot-1
    elif n == ntot:
        return 0
    else:
        return n


@jit(nopython=True, nogil=True, fastmath=True)
def get_back(queue, queue_index):
    """
    Return last grid point from queue, reset queue value, and reduce index.
    """
    t = queue[queue_index, 0]
    i = queue[queue_index, 1]
    j = queue[queue_index, 2]

    # Reset index (not really necessary, but easier for debugging..)
    queue[queue_index, 0] = 0
    queue[queue_index, 1] = 0
    queue[queue_index, 2] = 0

    queue_index -= 1

    return t, i, j, queue_index


@jit(nopython=True, nogil=True, fastmath=True)
def add_to_queue(t, i, j, queue, queue_index):
    """
    Advance index, and add grid point to queue.
    """
    queue_index += 1
    queue[queue_index, 0] = t
    queue[queue_index, 1] = i
    queue[queue_index, 2] = j

    return queue_index


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
def expand_region(
        unique_id, current_id, mask, base, top,
        queue, queue_index, nt, itot, jtot, i, j, t, time_tracking):
    """
    Search surounding grid points around (i,j,t) for clouds
    """
    unique_id[t,j,i] = current_id

    for di in (-1, 1):
        ic = cyclic_lims(i+di, itot)
        if mask[t,j,ic] and unique_id[t,j,ic] == 0 and no_overlap(base[t,j,i], top[t,j,i], base[t,j,ic], top[t,j,ic]):
            queue_index = add_to_queue(t, ic, j, queue, queue_index)

    for dj in (-1, 1):
        jc = cyclic_lims(j+dj, jtot)
        if mask[t,jc,i] and unique_id[t,jc,i] == 0 and no_overlap(base[t,j,i], top[t,j,i], base[t,jc,i], top[t,jc,i]):
            queue_index = add_to_queue(t, i, jc, queue, queue_index)

    if time_tracking:
        for dt in (-1, 1):
            tc = t+dt
            if tc >= 0 and tc < nt:
                if mask[tc,j,i] and unique_id[tc,j,i] == 0 and no_overlap(base[t,j,i], top[t,j,i], base[tc,j,i], top[tc,j,i]):
                    queue_index = add_to_queue(tc, i, j, queue, queue_index)

    return queue_index


@jit(nopython=True, nogil=True, fastmath=True)
def find_cells_kernel(cloud_id, mask, cloud_base, cloud_top, queue, nt, itot, jtot, time_tracking):
    """
    Numba kernel to keep the Python stuff separated...
    """

    # Active cloud id:
    current_id = 1

    # Index in queue of last active element.
    queue_index = -1

    for t in range(nt):
        if not time_tracking:
            current_id = 1

        for j in range(jtot):
            for i in range(itot):
                if mask[t,j,i] and cloud_id[t,j,i] == 0:

                    queue_index = add_to_queue(t, i, j, queue, queue_index)

                    while queue_index >= 0:

                        # Get last grid point from queue.
                        tt, ii, jj, queue_index = get_back(queue, queue_index)

                        # Find not yet assigned clouds in +/- 1 grid-point/time stencil, and add to queue.
                        queue_index = expand_region(
                                cloud_id, current_id, mask, cloud_base, cloud_top,
                                queue, queue_index, nt, itot, jtot, ii, jj, tt, time_tracking)

                    # Done with current cloud, advance unique id.
                    current_id += 1


def find_cells(mask, cloud_base, cloud_top, nt, itot, jtot, time_tracking):
    """
    Function to identify/label continuous cloudy regions.
    """

    # Output array with labelled cloud regions.
    cloud_id = np.zeros_like(mask, dtype=np.uint32)

    # Queue with connected but not-yet analysed grid points.
    queue = np.zeros((QUEUE_SIZE, 3), dtype=np.uint16)  # (t,i,j)

    # Loop over all dimensions to find connected regions.
    find_cells_kernel(cloud_id, mask, cloud_base, cloud_top, queue, nt, itot, jtot, time_tracking) 

    return cloud_id


def get_sizes(cloud_ids, A_gridpoint, nt, splitting=False):
    """
    Create array with all cloud/core sizes from all time steps,
    and calculate some statistics like clouds/time step, mean size, etc.
    """

    # Time statistics (number of clouds, size, ...)
    n_clouds = np.zeros(nt, dtype=np.int32)
    mean_cloud_size = np.zeros(nt, dtype=np.float32)

    for t in range(nt):
        n_clouds[t] = np.unique(cloud_ids[t,:,:]).size-1  # -1 for skipping id=0

    n_total_clouds = np.sum(n_clouds)
    cloud_sizes = np.zeros(n_total_clouds)

    # Get list with all cloud sizes:
    i = 0
    for t in range(nt):
        ids, counts = np.unique(cloud_ids[t,:,:], return_counts=True)
        sizes = np.sqrt(counts * A_gridpoint)
        cloud_sizes[i:i+ids.size-1] = sizes[1:]          # skip id=0
        if sizes.size > 1:
            mean_cloud_size[t] = np.mean(sizes[1:])
        i += ids.size-1

    return cloud_sizes, n_clouds, mean_cloud_size


if __name__ == '__main__':

    # Time indices to process:
    t0 = 360
    t1 = 961
    time_tracking = True
    dtype = np.float32

    # Read the total (ice+water) water path, and
    # column max cloud core theta_v perturbation
    data_path = '/home/scratch2/meteo_data/MicroHH/fesstval/josien/20210627'

    def read_var(nc_file, name):
        """
        Read variable from NetCDF file, cast into correct datatype.
        This is not the fastest method, but it has basically zero
        memory overhead on top of the output array size.
        """
        ds = xr.open_dataset(nc_file)
        out = np.empty_like(ds[name][t0:t1], dtype=dtype)

        print('reading {0}, size in memory={1:.3f} GB'.format(name, out.nbytes/1024**3))

        for t in range(t0, t1):
            out[t-t0] = ds[name][t].values

        out = np.ma.masked_where(out>1e16, out)

        return out

    def read_dims(nc_file):
        """
        Read dimensions from NetCDF file, cast into correct datatype.
        """
        ds = xr.open_dataset(nc_file)
        return ds.x.values, ds.y.values, ds.time[t0:t1].values

    # Read variables. Skipped when using `-i` in ipython, and data is already in memory.
    if 'wp_mask' not in locals():
        thv  = read_var('{}/qlqicore_max_thv_prime.xy.nc'.format(data_path), 'qlqicore_max_thv_prime')
        wp   = read_var('{}/qlqipath.xy.nc'.format(data_path), 'qlqipath')
        base = read_var('{}/qlqibase.xy.nc'.format(data_path), 'qlqibase')
        top  = read_var('{}/qlqitop.xy.nc'.format(data_path), 'qlqitop')

        # Read dimensions
        x,y,time = read_dims('{}/qlqitop.xy.nc'.format(data_path))

        itot = x.size
        jtot = y.size
        nt = time.size

        dx = x[1] - x[0]
        dy = y[1] = y[0]

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
    print('Tracking clouds')
    cloud_id = find_cells(wp_mask, base, top, nt, itot, jtot, time_tracking)
    print('Tracking cores')
    core_id = find_cells(thv_mask, base, top, nt, itot, jtot, time_tracking)

    print('Calculating cloud sizes')
    cloud_sizes, cloud_count, mean_cloud_size = get_sizes(cloud_id, dx*dy, nt)
    print('Calculating core sizes')
    core_sizes, core_count, mean_core_size = get_sizes(core_id, dx*dy, nt)

    # Save statistics in NetCDF format
    if time_tracking:
        nc_names = ['cloud_tracking_time.nc', 'core_tracking_time.nc']
    else:
        nc_names = ['cloud_tracking_inst.nc', 'core_tracking_inst.nc']

    values     = [cloud_id, core_id]
    sizes      = [cloud_sizes, core_sizes]
    counts     = [cloud_count, core_count]
    mean_sizes = [mean_cloud_size, mean_core_size]

    for n in range(2):
        nc_name = nc_names[n]
        ids = values[n]
        size = sizes[n]
        count = counts[n]
        mean_size = mean_sizes[n]

        nc = nc4.Dataset(nc_name, 'w')

        dim_x = nc.createDimension('x', itot)
        dim_y = nc.createDimension('y', jtot)
        dim_t = nc.createDimension('time', nt)
        dim_s = nc.createDimension('n', size.size)

        var_x = nc.createVariable('x', np.float32, 'x')
        var_y = nc.createVariable('y', np.float32, 'y')
        var_t = nc.createVariable('time', np.float32, 'time')

        var_id = nc.createVariable('id', np.uint32, ('time','y','x'))
        var_s  = nc.createVariable('size', np.float32, ('n'))

        var_nc = nc.createVariable('n_clouds', np.uint32, ('time'))
        var_size = nc.createVariable('mean_size', np.uint32, ('time'))

        var_x.setncatts({'units': 'm', 'long_name': 'grid point center x-direction'})
        var_y.setncatts({'units': 'm', 'long_name': 'grid point center y-direction'})
        var_t.setncatts({'units': 's', 'long_name': 'seconds since start of experiment'})
        var_id.setncatts({'units': '-', 'long_name': 'unique cloud ID'})
        var_s.setncatts({'units': 'm', 'long_name': 'unique cloud size as sqrt(area)'})
        var_nc.setncatts({'units': '-', 'long_name': 'cloud count'})
        var_size.setncatts({'units': 'm', 'long_name': 'mean cloud size'})

        var_x [:] = x
        var_y [:] = y
        var_t [:] = time
        var_id[:] = ids
        var_s [:] = size
        var_nc[:] = count
        var_size[:] = mean_size

        nc.close()
