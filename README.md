# Packaging C++ extension in Python using pybind11

This is a summary of the commands used in [the tutorial](tutorial.md).
To clone the repository and the submodules, run
```bash
git clone --recursive https://github.com/xflash96/pybind11_package_example
```

## The Docker developing environment
* To build the docker image, run `cd docker && ./build.sh`
* To run the container, `./run.sh` .

See the `docker/Dockerfile` for the details.

## Building the extension and installing the package
* To build the C++ extension, run `python setup.py build_ext -i`
* To install the package, run `sudo `which python` setup.py develop`

## Profiling
### [line_profiler](https://github.com/pyutils/line_profiler) for the Python code
* Mark the `@profile` to the function of interest. 
* Run the profiler via `kernprof -l ./bin/example_cmd`
* See the report with `python -m line_profiler example_cmd.lprof`

### Perf for the C++ extension
* Enable perf in Linux by `sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'`
* Profile via `perf record bin/example_cmd`
* Show report with `perf report`

### Test
Use either
* `python setup.py test`
* `pytest`
```
to run the unit tests.

### Releasing the package
* Pack the source distribution with `python setup.py sdist`
* Upload the package with `python -m twine upload --repository testpypi dist/*`
