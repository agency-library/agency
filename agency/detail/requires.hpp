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
#define __AGENCY_REQUIRES(...) typename std::enable_if<(__VA_ARGS__)>::type* = nullptr

