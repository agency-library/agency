#pragma once

#include <agency/detail/config.hpp>
#include <agency/executor/executor_traits.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/coordinate.hpp>
#include <agency/detail/type_traits.hpp>
#include <utility>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


template<class T, class Executor>
using shared_parameter_container = typename executor_traits<Executor>::template container<T>;


template<class Executor, class T>
shared_parameter_container<T,Executor> make_shared_parameter_container(Executor&, size_t n, const T& shared_init)
{
  return shared_parameter_container<T,Executor>(n, shared_init);
}


template<size_t depth, class Shape>
__AGENCY_ANNOTATION
size_t number_of_groups_at_depth(const Shape& shape)
{
  // to compute the number of groups at a particular depth given a shape,
  // take the first depth elements of shape and return shape_size
  return detail::shape_size(detail::tuple_take<depth>(shape));
}


template<class Executor, class... Types>
using tuple_of_shared_parameter_containers = detail::tuple<shared_parameter_container<Types,Executor>...>;

template<class Executor, class... Types>
struct tuple_of_shared_parameter_containers_war_nvbug1665680
{
  using type = detail::tuple<shared_parameter_container<Types,Executor>...>;
};


template<size_t... Indices, class Executor, class... Types>
// XXX WAR nvbug 1665680
//tuple_of_shared_parameter_containers<Executor,Types...>
typename tuple_of_shared_parameter_containers_war_nvbug1665680<Executor,Types...>::type
  make_tuple_of_shared_parameter_containers(detail::index_sequence<Indices...>, Executor& ex, typename executor_traits<Executor>::shape_type shape, const Types&... shared_inits)
{
  return detail::make_tuple(make_shared_parameter_container(ex, number_of_groups_at_depth<Indices>(shape), shared_inits)...);
}


template<class Executor, class... Factories>
// XXX WAR nvbug 1665680
//tuple_of_shared_parameter_containers<Executor, detail::result_of_t<Factories()>...>
typename tuple_of_shared_parameter_containers_war_nvbug1665680<Executor,detail::result_of_t<Factories()>...>::type
  make_tuple_of_shared_parameter_containers(Executor& ex, typename executor_traits<Executor>::shape_type shape, Factories... shared_factories)
{
  return make_tuple_of_shared_parameter_containers(detail::make_index_sequence<sizeof...(shared_factories)>(), ex, shape, shared_factories()...);
}


} // end executor_traits_detail
} // end detail
} // end agency

