# Box_Counting in python
Some codes to count particles in boxes and calculate statistics of these fluctuating counts

# Installation
To install, clone this directory into your site-packages directory (eg `cd ~/.local/lib/python3.10/site-packages && git clone https://github.com/brennansprinkle/Box_Counting countoscope`)
or simply download the codes

# Content
* example_runcounting.py              * example code to run the counting algorithm and plot relevant curves
* test_data/example_dataset.txt       * example data set simulated from Brownian motion of non-interacting particles
* Box_Count_Stats.py                  * source codes to count particles in boxes and calculate statistics
* Numba_Box_Count_stats.py            * same as above but with just-in-time-compilation. Sometimes numba is not compatible with some machines.
* LICENSE                             * licence agreement
* Old_Codes/                          * old codes directory

# Use
To use
```py
import Box_Counting as countoscope

results = countoscope.calculate_nmsd(data=f"data.dat", window_size_x=217.6, window_size_y=174, box_sizes=Box_Ls, sep_sizes=sep)
```
See the full example in `example_runcounting.py` (which also includes plotting)

The parameters to `calculate_nmsd` are:
* `data`. Either:
  * `data` should be a string, the address of a text file containing rows of x, y, t values, whitespace separated. 
  * or `data` should be provided as a 2D array where `data[i, 0:2]` is `[x, y, t]`
  * the `t` values should be the index of the frame. They can be 0-based or 1-based (or any other), and can be supplied as floats or integers.
* `window_size_x` and `window_size_y` optional, the dimensions of the viewing window. If not supplied, the maximum x and y coordinate over all frames and particles will be used instead.
  * it is assumed that the particles lie in `0 <= x <= window_size_x` and `0 <= y <= window_size_y`. Viewing windows not cornered at the origin are not currently supported.
* The box sizes are specified as:
  * if only `box_sizes` is provided, the boxes will be square, width and height of box `i` equal to `box_sizes[i]`.
  * if `box_sizes_x` and `box_sizes_y` are provided, box `i` will be of shape `box_sizes_x[i]` * `box_sizes_y[i]`.
  * however, if you want one of the width or height to be constant, you can just pass a single value to one of `box_sizes_x` or `box_sizes_y` and then all boxes will have the same width or height.
* `sep_sizes` should be an array of the same size as `box_sizes`/`box_sizes_x`/`box_sizes_y`
  * if any elements of `sep_sizes` are negative, the boxes will overlap. This causes the library to use a different algorithm to count the particles which is substantially slower. You should be careful when choosing the overlaps; if the overlap is a rational fraction of the box size then some boxes' edges will touch, leaving the counts correlated.

The return object:

| Property                | Shape                                                      | Description                                                                             |
|-------------------------|------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `results.N2_mean`       | len(box_sizes) * Nframes                                   | number displacement fluctuations (N(t) - N(0))^2 mean value                             |
| `results.N2_std`        | len(box_sizes) * Nframes                                   | number displacement fluctuations (N(t) - N(0))^2 mean value                             |
| `results.N_mean`        | len(box_sizes)                                             | mean particles per box, averaged over all boxes                                         |
| `results.N_mean_std`    | len(box_sizes)                                             | standard deviation of (mean particles per box in time) over all boxes                   |
| `results.N_var`         | len(box_sizes)                                             | variance of all counts                                                                  |
| `results.N_var_sem_ub`  | len(box_sizes)                                             |                                                                                         |
| `results.N_var_sem_lb`  | len(box_sizes)                                             |                                                                                         |
| `results.N_var_mod`     | len(box_sizes)                                             | variance of number of particles in box over time, averaged over all boxes               |
| `results.N_var_mod_std` | len(box_sizes)                                             | standard deviation of (variance of number of particles in box over time) over all boxes |
| `results.num_boxes`     | len(box_sizes)                                             | number of boxes used for each box size                                                  |
| `results.counts`        | len(box_sizes) * max_boxes_y * max_boxes_x * num_timesteps | the raw counts in each box                                                              |
| `results.box_coords`    | len(box_sizes) * max_boxes_y * max_boxes_x * 2             | the (x, y) positions of the lower-left corner of each box                               |


# Dependencies
numpy, scipy, and numba. We could make a version without numba, but I'm not sure why we would. By default, numba will run using as many threads as you have cores on your machine. Use `NUMBA_NUM_THREADS=16 python ....` to limit numba to a certain number of threads if you don't want it to use all your cores. If `tqdm` is installed, we will use it for nice progress bars.

# Timescale integral
The MATLAB file `timescale_integral.m` processes the data computed using `Fast_Box_Stats.py` by computing the timescale integral.

# Debugging
If you get an annoying C error like `Segmentation fault`, try running with `NUMBA_DISABLE_JIT=1 python myscript.py` to disable numba compilation (it will take much longer), and see if you get any pure Python errors