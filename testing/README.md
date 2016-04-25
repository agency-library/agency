This is the top-level directory of Agency's test programs.

# Building and Running Test Programs

To build the test programs, run the following command from this directory:

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

After setting up a SCons build environment, the `SConstruct` sets up a hierarchical build by invoking all subsidiary `SConscript` files in immediate child directories. A typical subsidiary
calls the `ProgramAndUnitTestPerSourceInCurrentDirectory()` method to turn each source file in its directory into a unit test program.

