/// \file
/// \brief Include this file to use any component of Agency which is not experimental and which requires
///        CUDA C++ language extensions.
///
/// Including `<agency/cuda.hpp>` recursively includes most of the header files organized beneath
/// `<agency/cuda/*>`. It is provided for quick access to most of Agency's CUDA features. Features not included
/// by this header, but which are organized beneath `<agency/cuda/*>` are considered experimental.
///
/// Specifically, `<agency/cuda.hpp>` provides definitions for all entities inside the `agency::cuda` namespace,
/// except for `agency::cuda::experimental`.
///

#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/algorithm.hpp>
#include <agency/cuda/container.hpp>
#include <agency/cuda/device.hpp>
#include <agency/cuda/future.hpp>
#include <agency/cuda/execution.hpp>
#include <agency/cuda/memory.hpp>

/// \namespace agency::cuda
/// \brief `agency::cuda` is the namespace which contains CUDA-specific functionality.
///

