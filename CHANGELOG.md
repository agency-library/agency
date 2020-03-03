Agency v0.3.0 Changelog
=======================

## Summary

Various changes driven by ISO C++ proposal P0443.

### Breaking Changes

  * `bulk_sync_execute` has been eliminated
  * `is_bulk_synchronous_executor` has been eliminated
  * `bulk_async_execute` has been eliminated
  * `is_bulk_asynchronous_executor` has been eliminated
  * `async_execute` has been eliminated
  * `is_asynchronous_executor` has been eliminated
  * `sync_execute` has been eliminated
  * `is_synchronous_executor` has been eliminated
  * `bulk_then_execute` has been eliminated
  * `is_bulk_continuation_executor` has been eliminated
  * `then_execute` has been eliminated
  * `is_continuation_executor` has been eliminated
  * `is_bulk_executor` has been eliminated
  * `is_simple_executor` has been eliminated
  * `future_value` and `future_value_t` have been renamed `future_result` and `future_result_t`, respectively.
  * `executor_execution_category` and `executor_execution_category_t` have been replaced with the `bulk_guarantee` executor property
  * `execution_categories.hpp` and the functional therein has been eliminated
  * `execution_agent_traits<A>::execution_category` has been replaced with `execution_agent_traits<A>::execution_requirement`
  * `cuda::deferred_future` has been eliminated

## New Features

  * `require` and properties:
    * `bulk`
    * `single`
    * `then`
    * `always_blocking`
    * `bulk_guarantee`
  * `basic_span`

### Containers

  * `cuda::vector`

### Executors

* Various executors now have equality operations.

### Utilities

* `pointer_adaptor`
* `cuda::device_ptr`
* `cuda::scoped_device`

## New Experimental Features

### Utilities

* `experimental::domain`
* `overload`


## Known Issues

  * [#437](../../issues/437) Nvcc emits warnings regarding standard library functions calling `__host__`-only functions

## Resolved Issues

  * [#428](../../issues/428) Warnings regarding ignored CUDA annotations have been eliminated


Agency v0.2.0 Changelog
=======================

## Summary

Agency 0.2.0 introduces new components for creating parallel C++ programs.  A
suite of new **containers** allow easy management of collections of objects in
parallel programs. `agency::array` and `agency::vector` provide familiar C++
components in CUDA codes while components like `shared` and `shared_vector`
allow groups of concurrent execution agents to cooperatively own an object.
New **executors** target CUDA cooperative kernels, OpenMP, loop unrolling, and
polymorphism. Finally, new speculative features allow programmers to experiment
with interfacing with native CUDA APIs, multidimensional arrays, and
range-based programming.

### Breaking Changes

  * `cuda::split_allocator` has been renamed `cuda::heterogeneous_allocator`

## New Features

### Control Structures

  * `async` : Composes with an executor to create a single asynchronous function invocation.
  * `invoke` : Composes with an executor to create a single synchronous function invocation.

### Containers
  * `array` : Statically-sized object container based on `std::array`.
  * `vector`: Dynamically-sized object container based on `std::vector`.
  * `shared`: Container for a single object shared by a group of concurrent execution agents.
  * `shared_array` : Container for a statically-sized collection of objects shared by a group of concurrent execution agents.
  * `shared_vector`: Container for a dynamically-sized collection of objects shared by a group of concurrent execution agents.
  * `tuple` : A product type based on `std::tuple`.

### Execution Policies

  * `concurrent_execution_policy_2d` : Induces a two-dimensional group of concurrent execution agents.
  * `sequenced_execution_policy_2d` : Induces a two-dimensional group of sequenced execution agents. 
  * `parallel_execution_policy_2d` : Induces a two-dimensional group of parallel execution agents.
  * `unsequenced_execution_policy_2d` : Induces a two-dimensional group of unsequenced execution agents.
  * OpenMP-specific execution policies
    * `omp::parallel_execution_policy` : Induces a group of parallel execution agents using an OpenMP executor.
    * `omp::unsequenced_execution_policy` : Induces a group of unsequenced execution agents using an OpenMP executor.

### Executors

  * `cuda::concurrent_grid_executor` : Creates concurrent-concurrent execution agents using CUDA 9's cooperative grid launch.
  * `omp::parallel_for_executor` AKA `omp::parallel_executor` : Creates parallel execution agents using OpenMP's parallel for loop.
  * `omp::simd_executor` AKA `omp::unsequenced_executor` : Creates unsequenced execution agents using OpenMP's SIMD for loop.
  * `experimental::unrolling_executor` : Creates sequenced execution agents using an unrolled for loop.
  * `variant_executor` : Creates execution agents with polymorphic execution guarantees using an dynamic, underlying executor.

### Utilities
  * `cuda::device` : Creates a `cuda::device_id` from a device enumerant.
  * `cuda::devices`: Creates a collection of `cuda::device_id` from a sequence of device enumerants.
  * `cuda::all_devices`: Creates a collection of `cuda::device_id` corresponding to all devices in the system.

## New Experimental Features

### Containers

  * `experimental::ndarray` : Dynamically-sized multidimensional object container.
  * `experimental::ndarray_ref`: View of a multidimensional object container.

### Execution Policies
  
  * `cuda::experimental::static_grid` : Induces a statically-sized group of parallel-concurrent execution agents using CUDA grid launch.
  * `cuda::experimental::static_con` : Indicues a statically-sized group of concurrent execution agents using CUDA grid launch.

### Utilities

  * `cuda::experimental::make_async_future` : Creates a `cuda::async_future` from underlying CUDA resources.
  * `cuda::experimental::make_dependent_stream` : Creates a `cudaStream_t` from a `cuda::async_future`.
  * Fancy ranges
    * `experimental::interval()` : Creates a range of integers specified by two end points.
    * `experimental::iota_view` : A range of increasing integers.
    * `experimental::transformed_view` : A transformed view of an underlying range.
    * `experimental::zip_with_view` : A zipped-and-then-transformed view of multiple underlying ranges.

## New Examples

  * [`fork_executor.cpp`](../0.2.0/examples/fork_executor.cpp)

## Known Issues

  * [#428](../../issues/428) nvcc 9.0 emits spurious warnings regarding ignored annotations

## Resolved Issues

  * [#347](../../issues/347) Various warnings at aggressive reporting levels have been eliminated
  * [#289](../../issues/289) `async_future::bulk_then()` needs to schedule the `outer_arg`'s destruction
  * [#352](../../issues/352) .rank() generates results in the wrong order

## Acknowledgments

  * Thanks to Steven Dalton and other Github users for submitting bug reports.
  * Thanks to Steven Dalton, Michael Garland, Mike Bauer, Isaac Gelado, Saurav Muralidharan, and Cris Cecka for continued input into Agency's overall design and implementation.


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

