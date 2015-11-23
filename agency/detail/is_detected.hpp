#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>

namespace agency
{
namespace detail
{
namespace is_detected_detail
{


template<class Default, class AlwaysVoid, template<class...> class Op, class... Args>
struct detector
{
  using value_t = std::false_type;
  using type = Default;
};
 
template<class Default, template<class...> class Op, class... Args>
struct detector<Default, void_t<Op<Args...>>, Op, Args...>
{
  using value_t = std::true_type;
  using type = Op<Args...>;
};


} // end is_detected_detail

struct nonesuch {};

template<template<class...> class Op, class... Args>
using is_detected = typename is_detected_detail::detector<nonesuch, void, Op, Args...>::value_t;
 
template<template<class...> class Op, class... Args>
using detected_t = typename is_detected_detail::detector<nonesuch, void, Op, Args...>::type;
 
template<class Default, template<class...> class Op, class... Args>
using detected_or = is_detected_detail::detector<Default, void, Op, Args...>;


} // end detail
} // end agency

