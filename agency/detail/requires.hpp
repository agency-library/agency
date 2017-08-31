#pragma once

#include <agency/detail/config.hpp>
#include <type_traits>

// The __AGENCY_REQUIRES() macro may be used in a function template's parameter list
// to simulate Concepts.
//
// For example, to selectively enable a function template only for integer types,
// we could do something like this:
//
//     template<class Integer,
//              __AGENCY_REQUIRES(std::is_integral<Integer>::value)
//             >
//     Integer plus_one(Integer x)
//     {
//       return x + 1;
//     }
//

#define __AGENCY_CONCATENATE_IMPL(x, y) x##y

#define __AGENCY_CONCATENATE(x, y) __AGENCY_CONCATENATE_IMPL(x, y)

#define __AGENCY_MAKE_UNIQUE(x) __AGENCY_CONCATENATE(x, __COUNTER__)

#define __AGENCY_REQUIRES_IMPL(unique_name, ...) bool unique_name = true, typename std::enable_if<(unique_name and __VA_ARGS__)>::type* = nullptr

#define __AGENCY_REQUIRES(...) __AGENCY_REQUIRES_IMPL(__AGENCY_MAKE_UNIQUE(__deduced_true), __VA_ARGS__)

