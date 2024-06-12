all: Py_Box_Count_Stats

Py_Box_Count_Stats: Py_Box_Count_Stats.cpp
	c++ -O3 -shared Py_Box_Count_Stats.cpp -o Py_Box_Count_Stats.so -std=c++14 -fPIC -I/usr/include/eigen3 -I/usr/include -lfftw3 `python3 -m pybind11 --includes`
clean:
	rm *.so
