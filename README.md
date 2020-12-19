# Packaging C++ extension in Python using pybind11

This is a summary of the commands used in [the tutorial](tutorial.md).
To clone the repository and the submodules, run
```bash
git clone --recursive https://github.com/xflash96/pybind11_package_example
```

## The Docker developing environment
To build the docker image, run
```bash
cd docker
./build.sh
```
To run the container, run
```bash
./run.sh
```
See the `docker/Dockerfile` for the details.

## Building the extension and installing the package
To build the C++ extension, run
```bash
python setup.py build_ext -i
```

To install the package, run
```bash
sudo `which python` setup.py develop
```

## Profiling
### [line_profiler](https://github.com/pyutils/line_profiler) for the Python code
Mark the `@profile` to the function of interest, and run
```bash
kernprof -l ./bin/example_cmd
```

To see the report, run
```bash
python -m line_profiler example_cmd.lprof
```

### Perf for the C++ extension
First, to enable perf in Linux, run
```bash
sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
```

Then profile the command of interest with
```bash
perf record bin/example_cmd
```
And show the report with
```bash
perf report
```

[line_profiler](https://github.com/pyutils/line_profiler)
```
kernprof -l bin/example_cmd
python -m line_profiler bin/example_cmd.lprof
```

### Test
Use either
```bash
python setup.py test # or
pytest
```
to run the unit tests.

### Releasing the package
Pack the source distribution with
```bash
python setup.py sdist
```
And upload it via
```bash
python -m twine upload --repository testpypi dist/*
```
