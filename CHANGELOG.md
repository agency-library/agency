Agency v0.2.0 Changelog
=======================

## Summary

TODO

### Breaking Changes

  * `cuda::split_allocator` has been renamed `cuda::heterogeneous_allocator`

## New Features

### New Control Structures

  * `async`

### New Utilities
  * `shared`
  * `shared_array`
  * `shared_vector`

### New Execution Policies

  * `concurrent_execution_policy_2d`
  * `sequenced_execution_policy_2d`
  * `parallel_execution_policy_2d`
  * `unsequenced_execution_policy_2d`
  * OpenMP-specific execution policies
    * `omp::parallel_execution_policy`
    * `omp::unsequenced_execution_policy`

### New Executors

  * `omp::parallel_for_executor` AKA `omp::parallel_executor`
  * `omp::simd_executor` AKA `omp::unsequenced_executor`
  * `experimental::unrolling_executor`

### New Experimental Execution Policies
  
  * `cuda::experimental::static_grid`
  * `cuda::experimental::static_con`

### New Experimental Utilities

  * `cuda::experimental::make_async_future`
  * `cuda::experimental::native_handle`
  * New fancy ranges
    * `experimental::interval()`
    * `experimental::iota_view`
    * `experimental::zip_with_view`

## Resolved Issues

  * [#347](../../issues/347) Various warnings at aggressive reporting levels have been eliminated
  * [#289](../../issues/289) `async_future::bulk_then()` needs to schedule the `outer_arg`'s destruction
  * [#352](../../issues/352) .rank() generates results in the wrong order


Agency v0.1.0 Changelog
=======================

## Summary

Agency 0.1.0 introduces new **control structures** such as `bulk_invoke()` for creating parallel tasks. A suite of new **execution policies** compose with these control structures to require different kinds of semantic guarantees from the created tasks. A new library of **executors** controls the mapping of tasks onto underlying execution resources such as CPUs, GPUs, and collections of multiple GPUs. In addition to these basic components, this release also introduces experimental support for a collection of utility types useful for creating Agency programs.

## New Features

### New Control Structures

  * `bulk_invoke`
  * `bulk_async`
  * `bulk_then`

### New Execution Policies

  * `concurrent_execution_policy`
  * `sequenced_execution_policy`
  * `parallel_execution_policy`
  * `unsequenced_execution_policy`
  * CUDA-specific execution policies
    * `cuda::concurrent_execution_policy`
    * `cuda::parallel_execution_policy`
    * `cuda::grid`

### New Executors

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
    * `cuda::concurrent_executor`
    * `cuda::grid_executor`
    * `cuda::grid_executor_2d`
    * `cuda::multidevice_executor`
    * `cuda::parallel_executor`

### New Experimental Utilities

  * `experimental::array`
  * `experimental::bounded_integer`
  * `experimental::optional`
  * `experimental::short_vector`
  * `experimental::span`
  * Fancy ranges based on the [range-v3](http://github.com/ericniebler/range-v3) library
    * `experimental::chunk_view`
    * `experimental::counted_view`
    * `experimental::stride_view`
    * `experimental::zip_view`

### New Examples

  * [`concurrent_ping_pong.cpp`](../0.1.0/examples/concurrent_ping_pong.cpp)
  * [`concurrent_sum.cpp`](../0.1.0/examples/concurrent_sum.cpp)
  * [`fill.cpp`](../0.1.0/examples/fill.cpp)
  * [`hello_async.cpp`](../0.1.0/examples/hello_async.cpp)
  * [`hello_lambda.cpp`](../0.1.0/examples/hello_lambda.cpp)
  * [`hello_then.cpp`](../0.1.0/examples/hello_then.cpp)
  * [`hello_world.cpp`](../0.1.0/examples/hello_world.cpp)
  * [`ping_pong_tournament.cpp`](../0.1.0/examples/ping_pong_tournament.cpp)
  * [`saxpy.cpp`](../0.1.0/examples/saxpy.cpp)
  * [`version.cpp`](../0.1.0/examples/version.cpp)
  * CUDA-specific example programs
    * [`async_reduce.cu`](../0.1.0/examples/cuda/async_reduce.cu)
    * [`black_scholes.cu`](../0.1.0/examples/cuda/black_scholes.cu)
    * [`hello_device_lambda.cu`](../0.1.0/examples/cuda/hello_device_lambda.cu)
    * [`multigpu_saxpy.cu`](../0.1.0/examples/cuda/multigpu_saxpy.cu)
    * [`saxpy.cu`](../0.1.0/examples/cuda/saxpy.cu)
    * [`simple_on.cu`](../0.1.0/examples/cuda/simple_on.cu)
    * [`transpose.cu`](../0.1.0/examples/cuda/transpose.cu)

## Known Issues

  * [#255](../../issues/255) Agency is not known to work with any version of the Microsoft Compiler
  * [#256](../../issues/256) Agency is not known to work with NVIDIA Compiler versions prior to 8.0
  * [#257](../../issues/257) Agency is not known to work with NVIDIA GPU architectures prior to `sm_3x`

## Acknowledgments

  * Thanks to Michael Garland for significant input into Agency's overall design.
  * Thanks to the NVIDIA compiler team, especially Jaydeep Marathe, for enhancements to `nvcc`'s C++ support.
  * Thanks to Steven Dalton, Mark Harris, and Evghenni Gaburov for testing this release during development.
  * Thanks to Duane Merrill and Sean Baxter for design feedback.
  * Thanks to Olivier Giroux for contributing an implementation of synchronic.


Agency v0.0.0 Changelog
=======================

## Summary

This version of Agency was not released.

