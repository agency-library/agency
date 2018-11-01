/// \file
/// \brief Include this file to use any component of Agency which is not experimental,
///        nor requires C++ language extensions.
///
/// Including `<agency/agency.hpp>` recursively includes most of Agency's header files. It is provided for quick and easy
/// access to most of Agency's features. Features not included by this header are either experimental,
/// or require C++ language extensions which may not be supported by all compilers (such as CUDA C++ language
/// extensions).
///
/// Individual header files provide finer-grained access to features. Using these individual header files may
/// result in shorter compilation times.
///

#pragma once

#include <agency/detail/config.hpp>
#include <agency/async.hpp>
#include <agency/bulk_async.hpp>
#include <agency/bulk_invoke.hpp>
#include <agency/bulk_then.hpp>
#include <agency/container.hpp>
#include <agency/coordinate.hpp>
#include <agency/exception_list.hpp>
#include <agency/execution.hpp>
#include <agency/functional.hpp>
#include <agency/future.hpp>
#include <agency/shared.hpp>
#include <agency/version.hpp>

/// \namespace agency
/// \brief `agency` is the top-level namespace which contains all Agency functionality.
///

