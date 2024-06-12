# Box_Counting -- old codes Readme

# C++ box counting
To run the code with a linux OS, first navigate to the working directory in a treminal and compile the C++ module with `make`. If you're running the code on MAC_OS, then rename the file `Makefile` to `Makefile_Linux` and the file `Makefile_MACOS` to `Makefile`.
The C++ module has dependencies:
* pybind11 (install using eg `pip install pybind11`)
* Eigen (download from https://eigen.tuxfamily.org/ and add to include path)
* FFTW (download from https://www.fftw.org/download.html and add to include path)

# Pure python box counting
To run the pure python code, simply modify the main `Fast_Box_Stats_NoCpp.py` and run with `python Fast_Box_Stats_NoCpp.py`

# Timescale integral
The MATLAB file `timescale_integral.m` processes the data computed using `Fast_Box_Stats.py` by computing the timescale integral.

