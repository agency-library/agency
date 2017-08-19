#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <utility>
#include <tuple> // #include this so that at least some definition of std::get<i>() exists
#include <type_traits>


namespace agency
{
namespace detail
{
namespace tuple_detail
{


template<class Reference, std::size_t i>
struct has_get_member_function_impl
{
  // XXX consider requiring that Result matches the expected result,
  //     i.e. propagate_reference_t<Reference, std::tuple_element_t<i,T>>
  template<class T = Reference,
           std::size_t j = i,
           class Result = decltype(std::declval<T>().template get<j>())
          >
  static std::true_type test(int);

  static std::false_type test(...);

  using type = decltype(test(0));
};

template<class Reference, std::size_t i>
using has_get_member_function = typename has_get_member_function_impl<Reference,i>::type;


// this implementation of get<i>() uses .get<i>()
__agency_exec_check_disable__
template<size_t i, class Tuple,
         __AGENCY_REQUIRES(
           has_get_member_function<Tuple&&,i>::value
         )>
__AGENCY_ANNOTATION
auto get(Tuple&& t) ->
  decltype(std::forward<Tuple>(t).template get<i>())
{
  return std::forward<Tuple>(t).template get<i>();
}


template<class Reference, std::size_t i>
struct has_std_get_free_function_impl
{
  // XXX consider requiring that Result matches the expected result,
  //     i.e. propagate_reference_t<Reference, std::tuple_element_t<i,T>>
  template<class T = Reference,
           std::size_t j = i,
           class Result = decltype(std::get<j>(std::declval<T>()))
          >
  static std::true_type test(int);

  static std::false_type test(...);

  using type = decltype(test(0));
};

template<class T, std::size_t i>
using has_std_get_free_function = typename has_std_get_free_function_impl<T,i>::type;


// this implementation of get<i>() uses std::get<i>
__agency_exec_check_disable__
template<size_t i, class Tuple,
         __AGENCY_REQUIRES(
           !has_get_member_function<Tuple&&,i>::value and
           has_std_get_free_function<Tuple&&,i>::value
         )>
__AGENCY_ANNOTATION
auto get(Tuple&& t) ->
  decltype(std::get<i>(std::forward<Tuple>(t)))
{
  return std::get<i>(std::forward<Tuple>(t));
}


} // end tuple_detail
} // end detail
} // end agency

