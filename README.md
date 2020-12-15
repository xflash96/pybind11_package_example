# Example Python package with pybind Cpp extension

### Docker developing environment
```bash
cd docker
./build.sh
./run.sh
```

### Build extension and install package
```bash
python setup.py build_ext -i
```

```bash
sudo `which python` setup.py develop
```

### Profiling
```bash
sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
```

```bash
perf record bin/example_cmd
perf report
```
https://opensource.com/article/18/7/fun-perf-and-python

line_profiler (https://github.com/pyutils/line_profiler)
```
kernprof -l benchmarks.py
python -m line_profiler benchmark.py.lprof
```

### Test
```bash
python setup.py test
pytest
```


### Release package
```bash
python setup.py sdist
python -m twine upload --repository testpypi dist/*
python -m twine upload dist/*
```
https://www.benjack.io/2018/02/02/python-cpp-revisited.html

### Details about PyBind11
https://github.com/pybind/pybind11/issues/1201
https://github.com/tdegeus/pybind11_examples
https://github.com/pybind/pybind11/blob/master/include/pybind11/numpy.h
