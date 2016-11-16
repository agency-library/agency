/// \file
/// \brief Include this file to use any component of Agency which is not experimental and which requires
///        OpenMP C++ language extensions.
///
/// Including `<agency/omp.hpp>` recursively includes most of the header files organized beneath
/// `<agency/omp/*>`. It is provided for quick access to most of Agency's OpenMP features. Features not included
/// by this header, but which are organized beneath `<agency/omp/*>` are considered experimental.
///
/// Specifically, `<agency/omp.hpp>` provides definitions for all entities inside the `agency::omp` namespace,
/// except for `agency::omp::experimental`.
///

#pragma once

#include <agency/detail/config.hpp>
#include <agency/omp/execution.hpp>

/// \namespace agency::omp
/// \brief `agency::omp` is the namespace which contains OpenMP-specific functionality.
///

