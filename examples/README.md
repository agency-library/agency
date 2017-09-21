# Building and Running Example Programs

Each example program is built from a single source file. To build an example program by hand, compile a source file with a C++11 or better compiler. For example, the following command builds the `hello_world.cpp` source file from the `examples` directory:

    $ clang -I.. -std=c++11 -lstdc++ -pthread hello_world.cpp

Example programs which require special compiler features, such as language extensions, are organized into subdirectories. For example, the `/cuda` subdirectory contains example programs which require a C++ compiler supporting CUDA language extensions.

CUDA C++ source (`.cu` files) should be built with the NVIDIA compiler (`nvcc`). Include the `--expt-extended-lambda` option:

    $ nvcc -I.. -std=c++11 --expt-extended-lambda cuda/saxpy.cu

## Automated Builds

The example programs may be built automatically with [Scons](https://scons.org), which is a portable, Python-based build tool.

To build automatically, run the following command from this directory:

    $ scons

To accelerate the build process, run the following command to run 8 jobs in parallel:

    $ scons -j8

To build *and* run the example programs, specify `run_examples` as a command line argument:

    $ scons -j8 run_examples

To build all examples underneath a particular subdirectory, run `scons` with the path to the subdirectory of interest as a command line argument.

For example, the following command builds all of the example programs underneath the `cuda` subdirectory:

    $ scons cuda

Likewise, the following command will build *and* run the example programs underneath the `cuda` subdirectory:

    $ scons cuda/run_examples

# Build System Structure

The top-level directory named 'examples' contains a `SConstruct` and `SConscript` file. `SContruct` contains definitions of common functionality used by the rest of the build system. `SConscript` describes what targets the build process should build. 

After setting up a SCons build environment, the `SConstruct` sets up a hierarchical build by invoking the top-level `SConscript` files in the root directory.
The top-level `SConscript` file calls the `RecursivelyCreateProgramsAndUnitTestAliases()` method to recursively traverse the directory tree and create a program from each source file.

