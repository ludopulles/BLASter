# This Makefile compiles Cython
.POSIX:

.PHONY: cython cython-gdb venv-clean eigen3-clean

# if virtualenv directory venv exists then prefer venv/bin/python3 over system python3
VENV    := venv
PYTHONV := python3
PYTHON  = $(or $(wildcard $(VENV)/bin/$(PYTHONV)), $(shell command -v $(PYTHONV)), $(PYTHONV))


all: cython

clean:
	$(PYTHON) setup.py clean
	rm -rf build cysignals_crash_logs

# Run
cython:
	$(PYTHON) setup.py build_ext

# Debug
# The LD_PRELOAD trick is from https://github.com/grpc/grpc/issues/25819
cython-gdb:
	$(PYTHON) setup.py build_ext --cython-gdb


### Rules to install Eigen C++ template library for linear algebra locally

# Default version of Eigen C++ template library for linear algebra
EIGEN_VERSION := 3.4.0

eigen3:
	$(MAKE) eigen-$(EIGEN_VERSION)
	if [ -d eigen3 ]; then rm eigen3; fi
	ln -s eigen-$(EIGEN_VERSION) eigen3

eigen-%: eigen-%.tar.gz
	tar -xzvf $< >/dev/null

eigen-%.tar.gz:
	wget -nv "https://gitlab.com/libeigen/eigen/-/archive/$*/$@" -O "$@"

eigen3-clean:
	rm -rf eigen3 eigen-$(EIGEN_VERSION) eigen-$(EIGEN_VERSION).tar.gz



### Rules to create a virtual environment with up-to-date numpy and cython

# Default requirements (maybe set a specific version of numpy & cython?)
PIP_REQUIREMENTS := pip cython cysignals numpy setuptools matplotlib

venv:
	@if [ "$(VIRTUAL_ENV)" != "" ]; then echo "Active virtual environment detected. Please run 'deactivate' first!"; false; fi
	$(PYTHON) -m pip install --upgrade pip 2> /dev/null || echo "ERROR: Upgrading pip failed!"
	$(PYTHON) -m virtualenv $(VENV) || $(PYTHON) -m venv $(VENV) || echo "ERROR: creating virtual environment failed! Please (pip) install virtualenv or venv."
	-@rm activate 2>/dev/null
	ln -s $(VENV)/bin/activate .
	printf "#!/usr/bin/env bash\nOPENBLAS_NUM_THREADS=1 $(VENV)/bin/$(PYTHONV) \$$*\n" > ./$(PYTHONV)
	chmod +x ./$(PYTHONV)
	$(VENV)/bin/$(PYTHONV) -m pip install --upgrade $(PIP_REQUIREMENTS)
	@echo "=================================================================="
	@echo "=== NOTE: Please use 'source activate' to activate environment ==="
	@echo "===       Or use './python3' to use environment                ==="
	@echo "=================================================================="

venv-clean:
	@if [ "$(VIRTUAL_ENV)" != "" ]; then echo "Active virtual environment detected. Please run 'deactivate' first!"; false; fi
	rm -rf venv
