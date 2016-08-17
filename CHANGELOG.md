Agency v0.1.0
=============

# Summary

TODO

Agency v0.1.0 is the initial experimental release 
Initial release of experimental library.

# New Features

## New Control Structures

  * `bulk_invoke`
  * `bulk_async`
  * `bulk_then`

## New Execution Policies

  * `concurrent_execution_policy`
  * `sequenced_execution_policy`
  * `parallel_execution_policy`
  * `unsequenced_execution_policy`
  * CUDA-specific execution policies
    * `cuda::concurrent_execution_policy`
    * `cuda::parallel_execution_policy`
    * `cuda::grid`

## New Executors

  * `concurrent_executor`
  * `executor_array`
  * `flattened_executor`
  * `parallel_executor`
  * `scoped_executor`
  * `sequenced_executor`
  * `unsequenced_executor`
  * `vector_executor`
  * CUDA-specific executors
    * `cuda::block_executor`
    * `cuda::grid_executor`
    * `cuda::grid_executor_2d`
    * `cuda::multidevice_executor`

## New Experimental Types

  * `experimental::array`
  * `experimental::bounded_integer`
  * `experimental::optional`
  * `experimental::short_vector`
  * `experimental::span`
  * Fancy ranges based on [range-v3](http://github.com/ericniebler/range-v3)
    * `experimental::chunk_view`
    * `experimental::counted_view`
    * `experimental::stride_view`
    * `experimental::zip_view`

## New Examples

  * `concurrent_ping_pong.cpp`
  * `concurrent_sum.cpp`
  * `fill.cpp`
  * `hello_async.cpp`
  * `hello_lambda.cpp`
  * `hello_world.cpp`
  * `ping_pong_tournament.cpp`
  * `saxpy.cpp`
  * `version.cpp`
  * CUDA-specific example programs
    * `async_reduce.cu`
    * `black_scholes.cu`
    * `hello_device_lambda.cu`
    * `multigpu_saxpy.cu`
    * `saxpy.cu`
    * `transpose.cu`

# Known Issues

  * #255 Agency is not known to work with any version of the Microsoft Compiler
  * #256 Agency is not known to work with NVIDIA Compiler versions prior to 8.0
  * #257 Agency is not known to work with NVIDIA GPU architectures prior to `sm_3x`

# Acknowledgments

  * Thanks to Michael Garland for significant input into Agency's overall design.
  * Thanks to Steven Dalton and Mark Harris for testing this release during development.
  * Thanks to Evghenni Gaburov and Duane Merrill for design feedback.
  * Thanks to Olivier Giroux for contributing an implementation of synchronic.


Agency v0.0.0
=============

# Summary

This version of Agency was not released.

