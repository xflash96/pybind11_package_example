# Example Python package with pybind Cpp extension

### Docker developing environment
```bash
cd docker
./build.sh
./run.sh
```

### Build extension and install package
sudo `which python` setup.py develop

### Profiling
```bash
sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
```
```bash
perf record bin/example_cmd
perf report
```
https://opensource.com/article/18/7/fun-perf-and-python

### Release package
```bash
sudo `which python` setup.py sdist
python3 -m twine upload dist/*
```

### Details about PyBind11
https://github.com/pybind/pybind11/issues/1201
https://github.com/tdegeus/pybind11_examples
https://github.com/pybind/pybind11/blob/master/include/pybind11/numpy.h
