This is the top-level directory of Agency's test programs.

# Building and Running Test Programs

Each test program is built from a single source file. To build a test program by hand, compile a source file with a C++11 or better compiler. For example, the following command builds the `bulk_invoke/multiple_results.cpp` source file from the `testing` directory:

    $ clang -I.. -std=c++11 -lstdc++ -pthread bulk_invoke/multiple_results.cpp

CUDA C++ source (`.cu` files) should be built with the NVIDIA compiler (`nvcc`). Include the `--expt-extended-lambda` option:

    $ nvcc -I.. -std=c++11 --expt-extended-lambda bulk_invoke/cuda/multiple_results.cu

## Automated Builds

The test programs may be built automatically with [Scons](https://scons.org), which is a portable, Python-based build tool.

To build automatically, run the following command from this directory:

    $ scons

To accelerate the build process, run the following command to run 8 jobs in parallel:

    $ scons -j8

To build *and* run the test programs, specify `run_tests` as a command line argument:

    $ scons -j8 run_tests

To build all tests underneath a particular subdirectory, run `scons` with the path to the subdirectory of interest as a command line argument.

For example, the following command builds all of the test programs underneath the `executor_traits` subdirectory:

    $ scons executor_traits

Likewise, the following command will build *and* run the tests programs underneath the `executor_traits` subdirectory:

    $ scons executor_traits/run_tests

# Build System Structure

The top-level directory named 'testing' contains a single `SConstruct` file. This file contains definitions of common functionality used by the rest of the build system.

After setting up a SCons build environment, the `SConstruct` sets up a hierarchical build by invoking the top-level `SConscript` files in the root directory.
The top-level `SConscript` file calls the `RecursivelyCreateProgramsAndUnitTestAliases()` method to recursively traverse the directory tree and create a program and unit test from each source file.

