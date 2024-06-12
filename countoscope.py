import numpy as np
import scipy.stats as stats
import numba
import numba.typed
import warnings
import collections
import multiprocessing
import functools


# we use tqdm for nice progress bars if it is available
try:
    import tqdm
    progressbar = functools.partial(tqdm.tqdm, leave=False)
except ImportError:
    progressbar = lambda x, desc=None, total=None: x

# These two functions are from SE
# https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft
def autocorrFFT(x):
    N = len(x)
    F = np.fft.fft(x, n=2*N)  # 2*N because of zero-padding
    PSD = F * np.conjugate(F)
    res = np.fft.ifft(PSD)
    res = (res[:N]).real  # now we have the autocorrelation in convention B
    n = N * np.ones(N) - np.arange(0, N)  # divide res(m) by (N-m)
    return res / n  # this is the autocorrelation in convention A

@numba.njit(fastmath=True)
def msd_fft1d(r):
    r = r.astype(np.float64) # if the dtype is int, it will propagate through and we'll get the wrong answer

    N = len(r)
    D = np.square(r)
    D = np.append(D, 0)
    with numba.objmode(S2='float64[:]'):
        S2 = autocorrFFT(r) # we have to run in objmode cause numba does not support fft
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m-1] - D[N-m]
        S1[m] = Q / (N-m)
    return S1 - 2 * S2


def msd_matrix_slice(matrix_slice):
    MSDs = np.zeros(matrix_slice.shape, dtype=np.float64)

    for j in range(matrix_slice.shape[0]):
        MSDs[j, :] = msd_fft1d(matrix_slice[j, :])
    return MSDs
    
def msd_matrix(matrix):
    # calculates the MSDs over axis 2

    slices = [matrix[i, :, :] for i in range(matrix.shape[0])]
    
    # we used to do this parralelisation in numba, but in testing, I found it was faster with multiprocessing, idk why
    with multiprocessing.Pool(16) as pool:
        results = list(progressbar(pool.imap(msd_matrix_slice, slices), total=len(slices), desc='msd matrix'))

    MSDs = np.zeros(matrix.shape)

    for i, MSD in enumerate(results):
        MSDs[i, :, :] = MSD

    return MSDs

def processDataFile(filename):
    data = np.fromfile(filename, dtype=float, sep=' ')
    #all_data = np.loadtxt('data/0.34_EKRM_trajs.dat', delimiter=',', skiprows=1)
    data = data.reshape((-1, 4))

    # print(f'data size (file) {data.nbytes/1e9}GB')
    
    return processDataArray(data)

@numba.njit()
def processDataArray(data):
    # returns (Xs, Ys) where Xs, Ys are lists, and Xs[t]/Ys[t] is a list of the x/y coordinates at time tassert data[:, 2].min() == 1, f'data timesteps should (presently) be 1-based. The first timestep was {data[:, 2].min()}'
    t0 = int(data[:, 2].min())
    Nframes = int(data[:, 2].max()) + 1 - t0 # this works because max_t is actually max(t)+1, and Nframes should be max(t)+1 when zero based

    # storing the data by continually appending to lists is incredibly slow
    # so instead we use a numpy array
    # but first we need to find the maximum number of simultaneous particles
    num_points_at_time = np.zeros((Nframes), dtype='int')
    for line_i in range(data.shape[0]):
        values = data[line_i, :]
        t = round(values[2])
        num_points_at_time[t-t0] += 1
 
    # then we add the particle coordinates into the numpy arrays
    Xs = np.full((Nframes, num_points_at_time.max()), np.nan)
    Ys = np.full((Nframes, num_points_at_time.max()), np.nan)
    # we used to have these in lists of lists, then lists of numpy arrays, but now just one numpy array
    # this is much more memory efficient (unless you have a very uneven number of particles per timestep, but we shouldn't)

    num_points_at_time = np.zeros((Nframes), dtype='int')

    for line_i in range(data.shape[0]):
        values = data[line_i, :]
        
        x = values[0]
        y = values[1]
        t = round(values[2])

        p = num_points_at_time[t-t0]
        Xs[t-t0, p] = x
        Ys[t-t0, p] = y

        num_points_at_time[t-t0] += 1

    min_x = data[:, 0].min()
    max_x = data[:, 0].max()
    min_y = data[:, 1].min()
    max_y = data[:, 1].max()
    return Xs, Ys, min_x, max_x, min_y, max_y

@numba.njit(parallel=True)
def do_counting_at_boxsize(x, y, window_size_x, window_size_y, box_size_x, box_size_y, offset_x, offset_y, num_timesteps, shift_size_x, shift_size_y):
    assert offset_x >= 0, 'currently we only handle positive offsets'
    assert offset_y >= 0, 'currently we only handle positive offsets'

    # shift_size is L + sep (in this fn, sep >= 0)
    num_boxes_x = int(np.floor((window_size_x - offset_x) / shift_size_x))
    num_boxes_y = int(np.floor((window_size_y - offset_y) / shift_size_y))

    counts = np.zeros((num_boxes_x, num_boxes_y, num_timesteps), dtype='uint16')
    
    assert num_boxes_x > 0, "num_boxes_x was zero"
    assert num_boxes_y > 0, "num_boxes_y was zero"

    assert offset_x + num_boxes_x * shift_size_x <= window_size_x # make sure that the final box doesn't overlap the edge of the window
    assert offset_y + num_boxes_y * shift_size_y <= window_size_y
    
    for time_index in numba.prange(num_timesteps):
        xt = x[time_index, :]
        yt = y[time_index, :]
        valid_points = ~np.isnan(xt)
        num_points = np.sum(valid_points) # number of x,y points available at this timestep

        for i in range(num_points):
            # find target box
            if xt[i] - offset_x < 0 or yt[i] - offset_y < 0:
                continue # these are points close to the origin that fall before the first box, when there's an offset
            target_box_x = int(np.floor((xt[i] - offset_x) / shift_size_x))
            target_box_y = int(np.floor((yt[i] - offset_y) / shift_size_y))

            # if the particle is at the far x or y, there may not be a box covering that last little bit
            if target_box_x >= num_boxes_x:
                continue
            if target_box_y >= num_boxes_y:
                continue

            # discard points that are in the sep border around the edge of the box
            distance_into_box_x = np.fmod(xt[i] - offset_x, shift_size_x)
            distance_into_box_y = np.fmod(yt[i] - offset_y, shift_size_y)
            if np.abs(distance_into_box_x-0.5*shift_size_x) >= box_size_x/2.0:
                continue
            if np.abs(distance_into_box_y-0.5*shift_size_y) >= box_size_y/2.0:
                continue

            # add this particle to the stats
            counts[target_box_x, target_box_y, time_index] += 1

    sep_size_x = shift_size_x - box_size_x
    sep_size_y = shift_size_y - box_size_y
    assert sep_size_x >= 0
    assert sep_size_y >= 0
    box_xs = np.arange(0, num_boxes_x, dtype=np.float64) * shift_size_x + offset_x + sep_size_x/2 # dtype needed in case the list happens to work as ints, then numba will be unable to unify the array later
    box_ys = np.arange(0, num_boxes_y, dtype=np.float64) * shift_size_y + offset_y + sep_size_y/2
       
    return counts, box_xs, box_ys


@numba.njit
def mod_ceil(x):
    # rounds such that 
    # x.0  ->  x+1
    # x.y  ->  x+1
    # in the counting we need it for the case where the box size is an exact divisor of window size
    if x % 1 == 0:
        return x+1
    else:
        return np.ceil(x)

@numba.njit
def approx_lte(x, y):
    # floating-point safe version of x <= y
    return x <= y or bool(np.isclose(x, y))
@numba.njit
def approx_gte(x, y):
    # floating-point safe version of x >= y
    return x >= y or bool(np.isclose(x, y))

@numba.njit(parallel=True)
def do_counting_at_boxsize_new(x, y, window_size_x, window_size_y, box_size_x, box_size_y, offset_x, offset_y, num_timesteps, shift_size_x, shift_size_y):
    assert offset_x >= 0, 'currently we only handle positive offsets'
    assert offset_y >= 0, 'currently we only handle positive offsets'

    # shift_size is L + sep (in this fn, sep >= 0)
    num_boxes_x = int(mod_ceil((window_size_x - offset_x - box_size_x) / shift_size_x))
    num_boxes_y = int(mod_ceil((window_size_y - offset_y - box_size_y) / shift_size_y))
    
    assert num_boxes_x > 0, "num_boxes_x was zero"
    assert num_boxes_y > 0, "num_boxes_y was zero"

    counts = np.zeros((num_boxes_x, num_boxes_y, num_timesteps), dtype='uint16')

    assert approx_lte(offset_x + (num_boxes_x-1)*shift_size_x + 1*box_size_x, window_size_x) # make sure that the final box doesn't overlap the edge of the window
    assert approx_lte(offset_y + (num_boxes_y-1)*shift_size_y + 1*box_size_y, window_size_y)
    
    for time_index in numba.prange(num_timesteps):
        xt = x[time_index, :]
        yt = y[time_index, :]
        valid_points = ~np.isnan(xt)
        num_points = np.sum(valid_points) # number of x,y points available at this timestep

        for i in range(num_points):
            # find target box
            if xt[i] - offset_x < 0 or yt[i] - offset_y < 0:
                continue # these are points close to the origin that fall before the first box, when there's an offset
            target_box_x = int(np.floor((xt[i] - offset_x) / shift_size_x))
            target_box_y = int(np.floor((yt[i] - offset_y) / shift_size_y))

            # if the particle is at the far x or y, there may not be a box covering that last little bit
            if target_box_x >= num_boxes_x:
                continue
            if target_box_y >= num_boxes_y:
                continue

            # discard points that are in the sep border around the edge of the box
            # this border is actually only on the top and right sides though
            distance_into_box_x = np.fmod(xt[i] - offset_x, shift_size_x)
            distance_into_box_y = np.fmod(yt[i] - offset_y, shift_size_y)
            if distance_into_box_x >= box_size_x:
                continue
            if distance_into_box_y >= box_size_y:
                continue

            # add this particle to the stats
            counts[target_box_x, target_box_y, time_index] += 1

    box_xs = np.arange(0, num_boxes_x, dtype=np.float64) * shift_size_x + offset_x # dtype needed in case shift_size_x and offset happen to be int, then numba will be unable to unify the array later
    box_ys = np.arange(0, num_boxes_y, dtype=np.float64) * shift_size_y + offset_y
       
    return counts, box_xs, box_ys

@numba.njit(fastmath=True)
def count_boxes(x, y, window_size_x, window_size_y, box_sizes_x, box_sizes_y, sep_sizes, offset_xs, offset_ys, use_old_overlap):
    # offset_x and offset_y will offset the whole grid of boxes from the origin
    # TODO: if offset is so big to reduce the number of boxes we might have a problem

    if offset_xs is None:
        offset_xs = np.zeros_like(box_sizes_x)
    if offset_ys is None:
        offset_ys = np.zeros_like(box_sizes_y)
    assert offset_xs.shape == box_sizes_x.shape
    assert offset_ys.shape == box_sizes_y.shape

    CountMs = numba.typed.List()
    all_box_xs = numba.typed.List()
    all_box_ys = numba.typed.List()
    # each of the arrays that gets appended in here will have a different shape
    # which is why we use a list not an ndarray

    for box_index in range(len(box_sizes_x)):
        num_timesteps = len(x)
        
        overlap = sep_sizes[box_index] < 0 # do the boxes overlap?
        
        print("Counting boxes L =", box_sizes_x[box_index], "*", box_sizes_y[box_index], ", sep =", sep_sizes[box_index], "overlapped" if overlap else "")

        if overlap and (box_sizes_x[box_index] % np.abs(sep_sizes[box_index]) == 0 or box_sizes_y[box_index] % np.abs(sep_sizes[box_index]) == 0):
            print('Negative overlap is an exact divisor of box size. This will lead to correlated boxes.')
        
        assert num_timesteps > 0
        
        if overlap:

            if use_old_overlap:
                # if the boxes overlap we cannot use the original method (below)
                # so we use this method instead, which is perhaps 25 times slower
                SepSize_x = box_sizes_x[box_index] + sep_sizes[box_index]
                SepSize_y = box_sizes_y[box_index] + sep_sizes[box_index]
                num_boxes_x = int(np.ceil((window_size_x - offset_xs[box_index] - box_sizes_x[box_index]) / SepSize_x))
                num_boxes_y = int(np.ceil((window_size_y - offset_ys[box_index] - box_sizes_y[box_index]) / SepSize_y))
                Counts = np.zeros((num_boxes_x, num_boxes_y, num_timesteps), dtype=np.uint16)

                box_xs = np.arange(0, num_boxes_x, dtype=np.float64) * SepSize_x + offset_xs[box_index] # dtype needed in case the arange happens to work as ints, then numba will be unable to unify the array later
                box_ys = np.arange(0, num_boxes_y, dtype=np.float64) * SepSize_y + offset_ys[box_index]

                for time_index in numba.prange(num_timesteps):
                    xt = x[time_index, :]
                    yt = y[time_index, :]
                    valid_points = ~np.isnan(xt)
                    num_points = np.sum(valid_points) # number of x,y points available at this timestep

                    for box_x_index in range(num_boxes_x):
                        for box_y_index in range(num_boxes_y):

                            box_x_min = box_xs[box_x_index]
                            box_x_max = box_x_min + box_sizes_x[box_index]
                            box_y_min = box_ys[box_y_index]
                            box_y_max = box_y_min + box_sizes_y[box_index]

                            assert box_x_max + sep_sizes[box_index]/2 < window_size_x
                            assert box_y_max + sep_sizes[box_index]/2 < window_size_y

                            for point in range(num_points):
                                if box_x_min < xt[point] and xt[point] <= box_x_max and box_y_min < yt[point] and yt[point] <= box_y_max:
                                    Counts[box_x_index, box_y_index, time_index] += 1.0

            else:
                # here is a 1d example to explain the variable names
                #  ___________________
                #  | ------  ------  |
                #  |     ------      |
                #  |_________________|
                #    <---->            box_size
                #    <-->              global shift - distance between start of one box and start of next box, boxes may be counted in different iterations
                #    <------>          local shift - distance between start of one box and start of the next box that ends up *in the same iteration*
                #   in this example, num_iters = 2

                global_shift_x = box_sizes_x[box_index] + sep_sizes[box_index] # L+sep, sep < 0
                global_shift_y = box_sizes_y[box_index] + sep_sizes[box_index]

                num_x_iters = int(np.ceil(box_sizes_x[box_index] / global_shift_x))
                num_y_iters = int(np.ceil(box_sizes_y[box_index] / global_shift_y))

                local_shift_x = num_x_iters * global_shift_x
                local_shift_y = num_y_iters * global_shift_y

                total_num_boxes_x = int(mod_ceil((window_size_x - offset_xs[box_index] - box_sizes_x[box_index]) / global_shift_x))
                total_num_boxes_y = int(mod_ceil((window_size_y - offset_ys[box_index] - box_sizes_y[box_index]) / global_shift_y))
                Counts = np.zeros((total_num_boxes_x, total_num_boxes_y, num_timesteps), dtype=np.uint16)

                box_xs = np.full((total_num_boxes_x,), np.nan)
                box_ys = np.full((total_num_boxes_y,), np.nan)

                for x_shift_index in range(num_x_iters):
                    x_shift = global_shift_x * x_shift_index

                    this_offset_x = offset_xs[box_index] + x_shift
                    
                    for y_shift_index in range(num_y_iters):
                        y_shift = global_shift_y * y_shift_index
                        this_offset_y = offset_ys[box_index] + y_shift
                        
                        counts_temp, box_xs_temp, box_ys_temp = do_counting_at_boxsize_new(x, y, window_size_x, window_size_y, 
                            box_sizes_x[box_index], box_sizes_y[box_index], this_offset_x, this_offset_y, num_timesteps, 
                            local_shift_x, local_shift_y)
                        # print(counts_temp.shape, 'into', Counts[x_shift_index::num_x_iters, y_shift_index::num_y_iters, :].shape)
                        Counts[x_shift_index::num_x_iters, y_shift_index::num_y_iters, :] = counts_temp # now we have the counts, we have to kind of "inter-tile" them to get the (num boxes x) * (num_boxes y) * (num timesteps) shape in a sensible way
                        box_xs[x_shift_index::num_x_iters] = box_xs_temp
                        box_ys[y_shift_index::num_y_iters] = box_ys_temp
        
        else:
            if use_old_overlap:
                local_shift_x = box_sizes_x[box_index] + sep_sizes[box_index] # SepSize is (in non overlapped sense) L+sep, needs a proper name
                local_shift_y = box_sizes_y[box_index] + sep_sizes[box_index]
                Counts, box_xs, box_ys = do_counting_at_boxsize(x, y, window_size_x, window_size_y, box_sizes_x[box_index], box_sizes_y[box_index], offset_xs[box_index], offset_ys[box_index], num_timesteps, local_shift_x, local_shift_y)
            
            else:
                local_shift_x = box_sizes_x[box_index] + sep_sizes[box_index] # SepSize is (in non overlapped sense) L+sep, needs a proper name
                local_shift_y = box_sizes_y[box_index] + sep_sizes[box_index]
                offset_x = offset_xs[box_index] + sep_sizes[box_index]/2 # this is so with non-overlapped boxes, the first box is not at (0, 0)
                offset_y = offset_ys[box_index] + sep_sizes[box_index]/2 # for backward compatibility
                Counts, box_xs, box_ys = do_counting_at_boxsize_new(x, y, window_size_x, window_size_y, box_sizes_x[box_index], box_sizes_y[box_index], offset_x, offset_y, num_timesteps, local_shift_x, local_shift_y)

        CountMs.append(Counts)
        all_box_xs.append(box_xs)
        all_box_ys.append(box_ys)

    print("Done with counting")
    return CountMs, all_box_xs, all_box_ys

@numba.njit
# numba cannot handle `np.var(counts, axis=2)` so we write it ourselves
def numba_var_3d_axis2(array):
    vars = np.zeros((array.shape[0], array.shape[1]))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            vars[i, j] = np.var(array[i, j, :])
    return vars

@numba.njit
# numba cannot handle `np.mean(counts, axis=2)` so we write it ourselves
def numba_mean_3d_axis2(array):
    means = np.zeros((array.shape[0], array.shape[1]))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            means[i, j] = np.mean(array[i, j, :])
    return means


@numba.njit(fastmath=True)
def computeMeanAndSecondMoment(counts):
    # counts is shape (num boxes x) * (num boxes y) * (num timesteps)
    mean_per_box = numba_mean_3d_axis2(counts)
    variance_per_box = numba_var_3d_axis2(counts)
    return mean_per_box.mean(), variance_per_box.mean(), mean_per_box.std(), variance_per_box.std()

def check_provided_box_sizes(box_sizes, box_sizes_x, box_sizes_y, sep_sizes):
    if box_sizes is not None:
        assert box_sizes_x is None and box_sizes_y is None, "if parameter box_sizes is provided, neither box_sizes_x nor box_sizes_y should be provided"
        box_sizes_x = box_sizes
        box_sizes_y = box_sizes
    else:
        assert box_sizes_x is not None and box_sizes_y is not None, "if box_sizes is not provided, both box_sizes_x and box_sizes_y should be provided"

        if np.isscalar(box_sizes_x):
            assert not np.isscalar(box_sizes_y), "if box_sizes_x is provided as a scalar, box_sizes_y should be an array"
            box_sizes_x = np.full_like(box_sizes_y, box_sizes_x)
        elif np.isscalar(box_sizes_y):
            assert not np.isscalar(box_sizes_x), "if box_sizes_y is provided as a scalar, box_sizes_x should be an array"
            box_sizes_y = np.full_like(box_sizes_x, box_sizes_y)
        
        assert len(box_sizes_x) == len(box_sizes_y)
        
    box_sizes_x = np.array(box_sizes_x) # ensure these are numpy arrays, not python lists or tuples
    box_sizes_y = np.array(box_sizes_y)
    
    assert np.all(~np.isnan(box_sizes_x)), "nan was found in box_sizes_x"
    assert np.all(~np.isnan(box_sizes_y)), "nan was found in box_sizes_y"
    assert np.all(~np.isnan(sep_sizes)),   "nan was found in sep_sizes"
    
    if np.isscalar(sep_sizes):
        sep_sizes = np.full_like(box_sizes_x, sep_sizes)
    else:
        assert len(box_sizes_x) == len(sep_sizes), "box_sizes(_x) and sep_sizes should have the same length"

    sep_sizes   = np.array(sep_sizes)
    assert np.all(- sep_sizes < box_sizes_x), '(-1) * sep_sizes[i] must always be smaller than box_sizes[i]'
    assert np.all(- sep_sizes < box_sizes_y)

    return box_sizes_x, box_sizes_y, sep_sizes

def load_data_and_check_window_size(data, window_size_x, window_size_y):
    # load data
    if type(data) is str:
        print("Reading data from file")
        Xs, Ys, min_x, max_x, min_y, max_y = processDataFile(data)
    else:
        assert np.all(~np.isnan(data)), "nan was found in data"
        print("Reading data from array")
        Xs, Ys, min_x, max_x, min_y, max_y = processDataArray(data)
    print("Done with data read")

    # check the window size is sensible
    if window_size_x is None:
        window_size_x = max_x
        print(f'Assuming window_size_x={window_size_x:.1f}')
    if window_size_y is None:
        window_size_y = max_y
        print(f'Assuming window_size_y={window_size_y:.1f}')
    # TODO: we haven't done anything if min_x is not zero

    assert min_x >= 0,             'An x-coordinate was supplied less than zero'
    assert max_x <= window_size_x, 'An x-coordinate was supplied greater than window_size_x'
    assert min_y >= 0,             'A y-coordinate was supplied less than zero'
    assert max_y <= window_size_y, 'A y-coordinate was supplied greater than window_size_x'

    warn_empty_thresh = 0.9
    if (max_x-min_x) < warn_empty_thresh * window_size_x:
        warnings.warn(f'x data fills less than {100*(max_x-min_x)/window_size_x:.0f}% of the window. Is window_size_x correct?')
    if (max_y-min_y) < warn_empty_thresh * window_size_y:
        warnings.warn(f'y data fills less than {100*(max_y-min_y)/window_size_y:.0f}% of the window. Is window_size_y correct?')

    return Xs, Ys, window_size_x, window_size_y

def calculate_nmsd(data, sep_sizes, window_size_x=None, window_size_y=None, box_sizes=None, box_sizes_x=None, box_sizes_y=None, offset_xs=None, offset_ys=None, use_old_overlap=False, return_counts=False):
    print(f'Will use {numba.get_num_threads()} threads')

    # input parameter processing
    box_sizes_x, box_sizes_y, sep_sizes = check_provided_box_sizes(box_sizes, box_sizes_x, box_sizes_y, sep_sizes)
    
    # load the data and check it
    Xs, Ys, window_size_x, window_size_y = load_data_and_check_window_size(data, window_size_x, window_size_y)
    del data # don't need this any more (it could be a big array in ram)

    # offset param checking, should do more!
    if offset_xs == None:
        offset_xs = np.zeros_like(box_sizes_x)
    if offset_ys == None:
        offset_ys = np.zeros_like(box_sizes_x)
    offset_xs = np.array(offset_xs)
    offset_ys = np.array(offset_ys)
    assert offset_xs.shape == box_sizes_x.shape
    assert offset_ys.shape == box_sizes_x.shape

    assert np.all(box_sizes_x < window_size_x), "None of box_sizes(_x) can be bigger than window_size_x"
    assert np.all(box_sizes_y < window_size_y), "None of box_sizes(_y) can be bigger than window_size_y"
    assert np.all(sep_sizes < window_size_y), "None of sep_sizes can be bigger than window_size_x"
    assert np.all(sep_sizes < window_size_y), "None of sep_sizes can be bigger than window_size_y"

    # now do the actual counting
    print("Compiling fast counting function")
    CountMs, box_xs, box_ys = count_boxes(Xs, Ys, window_size_x=window_size_x, window_size_y=window_size_y,
                          box_sizes_x=box_sizes_x, box_sizes_y=box_sizes_y, sep_sizes=sep_sizes,
                          offset_xs=offset_xs, offset_ys=offset_ys, use_old_overlap=use_old_overlap)
    
    num_timesteps = Xs.shape[0]
    del Xs, Ys # don't need these any more so save some RAM

    num_box_sizes = len(box_sizes_x)

    MSD_means = np.zeros((num_box_sizes, num_timesteps))
    MSD_stds  = np.zeros((num_box_sizes, num_timesteps))

    N_mean        = np.full(num_box_sizes, np.nan)
    N_var_mod     = np.full(num_box_sizes, np.nan)
    N_var_sem_lb  = np.full(num_box_sizes, np.nan)
    N_var_sem_ub  = np.full(num_box_sizes, np.nan)
    num_boxes     = np.full(num_box_sizes, np.nan)
    N_var         = np.full(num_box_sizes, np.nan)
    N_mean_std    = np.full(num_box_sizes, np.nan)
    N_var_mod_std = np.full(num_box_sizes, np.nan)

    print('Compiling MSD and mean functions')
    for box_index in range(num_box_sizes): # why isn't this a numba.prange?
        print("Processing boxes", box_index+1, "of", num_box_sizes)

        mean_N, variance, mean_N_std, variance_std = computeMeanAndSecondMoment(CountMs[box_index])
        variance_original = np.var(CountMs[box_index])

        alpha = 0.01
        df = 1.0 * CountMs[box_index].size - 1.0
        chi_lb = stats.chi2.ppf(0.5 * alpha, df)
        chi_ub = stats.chi2.ppf(1.0 - 0.5 * alpha, df)
        variance_sem_lb = (df / chi_lb) * variance
        variance_sem_ub = (df / chi_ub) * variance

        N_mean       [box_index] = mean_N
        N_mean_std   [box_index] = mean_N_std # after taking the mean particles in each box over time, this is the std.dev over all boxes
        N_var_mod    [box_index] = variance # this is the variance in counts for one box, averaged over all boxes
        N_var_mod_std[box_index] = variance_std
        N_var_sem_lb [box_index] = variance_sem_lb
        N_var_sem_ub [box_index] = variance_sem_ub
        num_boxes    [box_index] = CountMs[box_index].shape[0] * CountMs[box_index].shape[1] # number of boxes counted over
        N_var        [box_index] = variance_original # this is simply the variance over all counts
        
        MSDs = msd_matrix(CountMs[box_index])

        MSD_means[box_index, :] = np.mean(MSDs, axis=(0, 1))
        MSD_stds [box_index, :] = np.std (MSDs, axis=(0, 1))

    # we make versions of box_xs/box_ys padded with nan, so that it has a homogeneous shape, and can be a numpy array
    max_num_boxes_x = max([b.size for b in box_xs])
    max_num_boxes_y = max([b.size for b in box_ys])
    all_coords = np.full((num_box_sizes, max_num_boxes_y, max_num_boxes_x, 2), np.nan)
    for i in range(len(CountMs)):
        box_coords_x, box_coords_y = np.meshgrid(box_xs[i], box_ys[i])
        box_coords = np.stack([box_coords_x, box_coords_y], axis=2)
        all_coords[i, :box_ys[i].size, :box_xs[i].size, :] = box_coords
    
    if return_counts:
        print('Making counts array')
        # we make a new version of CountsMs padded with nan, so that it has a homogeneous shape, and can be a numpy array
        all_counts = np.full((num_box_sizes, max_num_boxes_x, max_num_boxes_y, CountMs[0].shape[2]), np.nan) # this data is actually uint16, but the array has to be float because you can't store nan in int arrays; only float
        for i in range(len(CountMs)):
            all_counts[i, :CountMs[i].shape[0], :CountMs[i].shape[1], :] = CountMs[i]
    else:
        all_counts = None

    Results = collections.namedtuple('Results', ['nmsd', 'nmsd_std', 'N_mean', 'N_var', 'N_var_sem_lb', 'N_var_sem_ub', 'num_boxes', 'N_var_mod', 'N_mean_std', 'N_var_mod_std', 'counts', 'box_coords'])
    return Results(nmsd=MSD_means, nmsd_std=MSD_stds, N_mean=N_mean, N_var_mod=N_var_mod,
                       N_var_sem_lb=N_var_sem_lb, N_var_sem_ub=N_var_sem_ub, num_boxes=num_boxes,
                       N_var=N_var, N_mean_std=N_mean_std, N_var_mod_std=N_var_mod_std,
                       counts=all_counts, box_coords=all_coords)
    # input parameter processing
    box_sizes_x, box_sizes_y, sep_sizes = check_provided_box_sizes(box_sizes, box_sizes_x, box_sizes_y, sep_sizes)
    
    # load the data and check it
    Xs, Ys, window_size_x, window_size_y = load_data_and_check_window_size(data, window_size_x, window_size_y)

    assert np.all(box_sizes_x < window_size_x), "None of box_sizes(_x) can be bigger than window_size_x"
    assert np.all(box_sizes_y < window_size_y), "None of box_sizes(_y) can be bigger than window_size_y"
    assert np.all(sep_sizes < window_size_y), "None of sep_sizes can be bigger than window_size_x"
    assert np.all(sep_sizes < window_size_y), "None of sep_sizes can be bigger than window_size_y"

    # now do the actual counting
    print("Compiling fast counting function (this may take a min. or so)")
    Xnb = numba.typed.List(np.array(xi) for xi in Xs) # TODO why is this defined here not inside count_boxes?
    Ynb = numba.typed.List(np.array(yi) for yi in Ys) # TODO why is this defined here not inside count_boxes?

    CountMs = count_boxes(Xnb, Ynb, window_size_x=window_size_x, window_size_y=window_size_y,
                                        box_sizes_x=box_sizes_x, box_sizes_y=box_sizes_y, sep_sizes=sep_sizes)

    N_Stats = np.zeros((len(box_sizes_x), 6))

    MSD_means = np.zeros((len(box_sizes_x), len(Xs)))
    MSD_stds  = np.zeros((len(box_sizes_x), len(Xs)))

    for box_index in range(len(box_sizes_x)):
        # why isn't this a numba.prange?
        print("Processing Box size:", box_sizes_x[box_index], "*", box_sizes_y[box_index])

        N_Stats[box_index, 0] = box_sizes_x[box_index]
        #mean, variance, variance_sem_lb, variance_sem_ub = computeMeanAndSecondMoment(CountMs[lbIdx])
        mean_N, variance = computeMeanAndSecondMoment(CountMs[box_index])

        ####################
        alpha = 0.01
        df = 1.0 * CountMs[box_index].size - 1.0
        chi_lb = stats.chi2.ppf(0.5 * alpha, df)
        chi_ub = stats.chi2.ppf(1.0 - 0.5 * alpha, df)
        variance_sem_lb = (df / chi_lb) * variance
        variance_sem_ub = (df / chi_ub) * variance
        ####################

        N_Stats[box_index, 1] = mean_N
        N_Stats[box_index, 2] = variance
        N_Stats[box_index, 3] = variance_sem_lb
        N_Stats[box_index, 4] = variance_sem_ub
        N_Stats[box_index, 5] = CountMs[box_index].shape[0] * CountMs[box_index].shape[1] # number of boxes counted over

        MSDs = msd_matrix(CountMs[box_index])

        MSD_means[box_index, :] = np.mean(MSDs, axis=(0, 1))
        MSD_stds [box_index, :] = np.std (MSDs, axis=(0, 1))

    return MSD_means, MSD_stds, N_Stats
