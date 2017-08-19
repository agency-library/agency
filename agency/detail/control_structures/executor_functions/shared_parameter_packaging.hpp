#pragma once

#include <agency/detail/tuple/tuple_utility.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/tuple.hpp>
#include <agency/detail/tuple_matrix.hpp>
#include <agency/detail/control_structures/bind.hpp>
#include <agency/detail/control_structures/shared_parameter.hpp>
#include <agency/detail/factory.hpp>
#include <type_traits>
#include <utility>
#include <functional>

namespace agency
{
namespace detail
{


// signifies an element of the shared parameter matrix which is not occupied
struct null_type {};


template<size_t I, class... Types>
struct find_exactly_one_not_null_impl;


template<size_t I, class... Types>
struct find_exactly_one_not_null_impl<I, null_type, Types...> : find_exactly_one_not_null_impl<I+1, Types...> {};


template<size_t I, class T, class... Types>
struct find_exactly_one_not_null_impl<I, T, Types...> : std::integral_constant<size_t, I>
{
  static_assert(find_exactly_one_not_null_impl<0,Types...>::value == sizeof...(Types), "non-null type can only occur once in type list");

  using type = T;
};


template<size_t I>
struct find_exactly_one_not_null_impl<I> : std::integral_constant<size_t, I> {};


// note that we decay the type when searching for null_type
template<class... Types>
struct find_exactly_one_not_null : find_exactly_one_not_null_impl<0, typename std::decay<Types>::type...>
{
  static_assert(find_exactly_one_not_null::value < sizeof...(Types), "type not found in type list");
};


template<class Tuple>
struct tuple_find_non_null_result;


template<class... Types>
struct tuple_find_non_null_result<tuple<Types...>>
  : std::add_lvalue_reference<
      typename find_exactly_one_not_null<Types...>::type
    >
{};


template<class... Types>
__AGENCY_ANNOTATION
typename tuple_find_non_null_result<tuple<Types...>>::type
  tuple_find_non_null(const tuple<Types...>& t)
{
  return agency::get<find_exactly_one_not_null<Types...>::value>(t);
}


// to package shared parameters for an executor,
// we create a shared parameter matrix
// the rows correspond to levels of the executor's hierarchy
// the columns correspond to shared arguments

// Example - Suppose we have an executor with three levels of execution hierarchy.
//
// For an invocation like:
//
//     bulk_invoke(executor, function, shape, share<1>(13), share<0>(7), share<0>(42));
//
// We have three shared parameters - 2 for level 0, 1 for level 1, and 0 for level 2.
// For these parameters, make_shared_parameter_matrix() will return the following matrix, encoded as a tuple of tuples:
//
// {{ null,    7,   42 },
//  {   13, null, null },
//  { null, null, null }}
//
// Each row of this matrix corresponds to a level of the executor's execution hierarchy. Non-null elements of row i are parameters that are shared by execution agents at level i of the execution hierarchy.
// Each column j of this matrix corresponds to a shared parameter; i.e., an element of shared_arg_tuple.
// Note that each column j of the matrix has exactly one non-null element -- the jth shared parameter.
//
// Since we send factories to an executor, instead of sending a matrix of values, we send a tuple factories.
// Each factory creates a row of the matrix.
//
// For the above example, the corresponding tuple of factories would look like this:
//
// {{ zip_factory<   null_factory, factory<int>{7}, factory<int>{42} },
//  { zip_factory<factory<int>{13},   null_factory,     null_factory },
//  { zip_factory<   null_factory,    null_factory,     null_factory }}
//
// To unpack this matrix into a tuple of parameters for the user's function, we need to find the non-null element of each column.
// Currently the most straightforward way to do this is to transpose the matrix and find the one non-null element of each row of the transpose.


struct null_factory
{
  __AGENCY_ANNOTATION
  null_type operator()() const
  {
    return null_type{};
  }
};


template<size_t level>
struct make_factory_tuple_element
{
  template<class Factory>
  __AGENCY_ANNOTATION
  Factory operator()(const agency::detail::shared_parameter<level,Factory>& parm) const
  {
    return parm.factory();
  }

  template<size_t other_level, class Factory>
  __AGENCY_ANNOTATION
  null_factory operator()(const agency::detail::shared_parameter<other_level,Factory>&) const
  {
    return null_factory{};
  }
};


template<size_t execution_level, class Tuple>
using shared_parameter_factory_t = decltype(
  make_zip_factory(agency::detail::tuple_map(make_factory_tuple_element<execution_level>{}, std::declval<Tuple>()))
);


template<size_t execution_level, class... SharedArgs>
__AGENCY_ANNOTATION
shared_parameter_factory_t<execution_level,agency::tuple<SharedArgs...>> make_shared_parameter_factory(const agency::tuple<SharedArgs...>& shared_arg_tuple)
{
  return make_zip_factory(agency::detail::tuple_map(make_factory_tuple_element<execution_level>{}, shared_arg_tuple));
}


template<class Indices, class Tuple>
struct shared_parameter_factory_tuple_t_impl;


template<size_t... ExecutionLevel, class Tuple>
struct shared_parameter_factory_tuple_t_impl<agency::detail::index_sequence<ExecutionLevel...>,Tuple>
{
  using type = agency::tuple<
    shared_parameter_factory_t<ExecutionLevel,Tuple>...
  >;
};


template<size_t execution_depth, class Tuple>
using shared_parameter_factory_tuple_t = typename shared_parameter_factory_tuple_t_impl<
  agency::detail::make_index_sequence<
    execution_depth
  >,
  Tuple
>::type;


template<size_t... ExecutionLevel, class... SharedArgs>
__AGENCY_ANNOTATION
shared_parameter_factory_tuple_t<sizeof...(ExecutionLevel), agency::tuple<SharedArgs...>>
  make_shared_parameter_factory_tuple_impl(agency::detail::index_sequence<ExecutionLevel...>,
                                           const agency::tuple<SharedArgs...>& shared_arg_tuple)
{
  return agency::make_tuple(
    make_shared_parameter_factory<ExecutionLevel>(shared_arg_tuple)...
  );
}


template<size_t executor_depth, class... SharedArgs>
__AGENCY_ANNOTATION
shared_parameter_factory_tuple_t<executor_depth,agency::tuple<SharedArgs...>>
  make_shared_parameter_factory_tuple(const agency::tuple<SharedArgs...>& shared_arg_tuple)
{
  return make_shared_parameter_factory_tuple_impl(
    agency::detail::make_index_sequence<executor_depth>{},
    shared_arg_tuple
  );
}


template<class RowIndices, class TupleMatrix>
struct extracted_shared_parameters_t_impl;


template<size_t... RowIndices, class TupleMatrix>
struct extracted_shared_parameters_t_impl<index_sequence<RowIndices...>,TupleMatrix>
{
  using type = agency::tuple<
    typename tuple_find_non_null_result<
      typename std::tuple_element<
        RowIndices,
        TupleMatrix
      >::type
    >::type...
  >;
};


template<class TupleMatrix>
using extracted_shared_parameters_t = typename extracted_shared_parameters_t_impl<
  make_index_sequence<
    tuple_matrix_shape<TupleMatrix>::rows
  >,
  TupleMatrix
>::type;


template<size_t... RowIndex, class... Rows>
__AGENCY_ANNOTATION
extracted_shared_parameters_t<tuple<Rows...>>
  extract_shared_parameters_from_rows_impl(index_sequence<RowIndex...>, const tuple<Rows...>& mtx)
{
  return agency::tie(
    detail::tuple_find_non_null(agency::get<RowIndex>(mtx))...
  );
}


template<class... Rows>
__AGENCY_ANNOTATION
extracted_shared_parameters_t<tuple<Rows...>>
  extract_shared_parameters_from_rows(const tuple<Rows...>& mtx)
{
  return extract_shared_parameters_from_rows_impl(
    make_index_sequence<
      sizeof...(Rows)
    >{},
    mtx
  );
}


template<class TupleMatrix>
using unpack_shared_parameters_from_executor_t = extracted_shared_parameters_t<tuple_matrix_transpose_view<TupleMatrix>>;


template<class... Rows>
__AGENCY_ANNOTATION
unpack_shared_parameters_from_executor_t<tuple<Rows&...>>
  unpack_shared_parameter_matrix_from_executor(tuple<Rows&...> shared_param_matrix)
{
  const size_t num_columns = tuple_matrix_shape<tuple<Rows&...>>::columns;

  // to transform the shared_param_matrix into a tuple of shared parameters
  // we need to find the actual (non-null) parameter in each column of the matrix
  // the easiest way to do this is to tranpose the matrix and extract the shared parameter from each row of the transpose
  return detail::extract_shared_parameters_from_rows(
    detail::make_transposed_view<num_columns>(
      shared_param_matrix
    )
  );
}


template<class... Types>
__AGENCY_ANNOTATION
unpack_shared_parameters_from_executor_t<tuple<Types&...>>
  unpack_shared_parameters_from_executor(Types&... shared_params)
{
  return detail::unpack_shared_parameter_matrix_from_executor(agency::tie(shared_params...));
}


template<class... Args>
__AGENCY_ANNOTATION
auto forward_shared_parameters_as_tuple(Args&&... args)
  -> decltype(
       __tu::tuple_filter_invoke<is_shared_parameter_ref>(
         agency::forward_as_tuple(std::forward<Args>(args)...),
         detail::forwarder()
       )
     )
{
  return __tu::tuple_filter_invoke<is_shared_parameter_ref>(
    agency::forward_as_tuple(std::forward<Args>(args)...),
    detail::forwarder()
  );
}


} // end detail
} // end agency

