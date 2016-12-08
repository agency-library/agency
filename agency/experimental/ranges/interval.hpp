#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/experimental/ranges/iota.hpp>

namespace agency
{
namespace experimental
{


template<class Arithmetic, class OtherArithmetic,
         __AGENCY_REQUIRES(
           std::is_arithmetic<Arithmetic>::value
         ),
         __AGENCY_REQUIRES(
           std::is_convertible<
             OtherArithmetic, Arithmetic
           >::value
         )>
auto interval(Arithmetic begin, OtherArithmetic end) ->
  decltype(iota(begin, end))
{
  return iota(begin, end);
}


} // end experimental
} // end agency

