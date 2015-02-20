#pragma once

#include <agency/detail/tuple_utility.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/tuple_matrix.hpp>
#include <agency/detail/bind.hpp>
#include <type_traits>
#include <utility>
#include <functional>

namespace agency
{
namespace detail
{


template<class... Types>
__AGENCY_ANNOTATION
const typename std::tuple_element<0,tuple<Types...>>::type&
  tuple_find_if_impl(const tuple<Types...>& filtered_tuple)
{
  return detail::get<0>(filtered_tuple);
}


template<template<class> class MetaFunction, class Tuple>
using tuple_find_if_t = decltype(
  tuple_find_if_impl(
    __tu::tuple_filter_invoke<MetaFunction>(
      std::declval<Tuple>(),
      std::declval<agency::detail::forwarder>()
    )
  )
);


template<template<class> class MetaFunction, class Tuple>
TUPLE_UTILITY_ANNOTATION
tuple_find_if_t<MetaFunction,Tuple>
  tuple_find_if(Tuple&& t)
{
  return tuple_find_if_impl(
    __tu::tuple_filter_invoke<MetaFunction>(
      std::forward<Tuple>(t),
      detail::forwarder{}
    )
  );
}


// signifies an element of the shared parameter matrix which is not occupied
struct null_type {};


// when looking for null_type, strip references first
template<class T>
struct is_null : std::is_same<null_type, typename std::decay<T>::type> {};


template<class T>
struct is_not_null : 
  std::integral_constant<
    bool,
    !is_null<T>::value
  >
{};


template<class... Types>
__AGENCY_ANNOTATION
tuple_find_if_t<is_not_null,tuple<Types...>>
  tuple_find_non_null(const tuple<Types...>& t)
{
  return detail::tuple_find_if<is_not_null>(t);
}


template<size_t level, class T, class... Args>
struct shared_parameter
{
  __AGENCY_ANNOTATION
  T make(std::integral_constant<size_t,level>) const
  {
    return __tu::make_from_tuple<T>(args_);
  }

  template<size_t other_level>
  __AGENCY_ANNOTATION
  null_type make(std::integral_constant<size_t,other_level>) const
  {
    return null_type{};
  }

  tuple<Args...> args_;
};


template<class T> struct is_shared_parameter : std::false_type {};
template<size_t level, class T, class... Args>
struct is_shared_parameter<shared_parameter<level,T,Args...>> : std::true_type {};


template<class T>
struct is_shared_parameter_ref
  : std::integral_constant<
      bool,
      (std::is_reference<T>::value && is_shared_parameter<typename std::remove_reference<T>::type>::value)
    >
{};


template<size_t level>
struct call_make
{
  template<size_t other_level, class T, class... Args>
  __AGENCY_ANNOTATION
  auto operator()(const shared_parameter<other_level,T,Args...>& parm) const
    -> decltype(
         parm.make(std::integral_constant<size_t,level>{})
       )
  {
    return parm.make(std::integral_constant<size_t,level>{});
  }
};


template<size_t column_idx, size_t... row_idx, class TupleMatrix>
__AGENCY_ANNOTATION
auto tuple_matrix_column_impl(index_sequence<row_idx...>, TupleMatrix&& mtx)
  -> decltype(
       detail::tie(
           detail::get<column_idx>(detail::get<row_idx>(std::forward<TupleMatrix>(mtx)))...
         )
     )
{
  return detail::tie(
    detail::get<column_idx>(detail::get<row_idx>(std::forward<TupleMatrix>(mtx)))...
  );
}


template<size_t column_idx, class TupleMatrix>
__AGENCY_ANNOTATION
auto tuple_matrix_column(TupleMatrix&& mtx)
  -> decltype(
       tuple_matrix_column_impl<column_idx>(
         make_index_sequence<
           std::tuple_size<
             typename std::decay<TupleMatrix>::type
           >::value
         >{},
         std::forward<TupleMatrix>(mtx)
       )
     )
{
  return tuple_matrix_column_impl<column_idx>(
    make_index_sequence<
      std::tuple_size<
        typename std::decay<TupleMatrix>::type
      >::value
    >{},
    std::forward<TupleMatrix>(mtx)
  );
}


template<class Indices, class Tuple>
struct packaged_shared_parameters_t_impl;


template<size_t... RowIndex, class Tuple>
struct packaged_shared_parameters_t_impl<index_sequence<RowIndex...>,Tuple>
{
  using type = decltype(
    detail::make_tuple(
      detail::tuple_map(detail::call_make<RowIndex>{}, std::declval<Tuple>())...
    )
  );
};


template<class Tuple>
using packaged_shared_parameters_t = typename packaged_shared_parameters_t_impl<
  agency::detail::make_index_sequence<
    std::tuple_size<Tuple>::value
  >,
  Tuple
>::type;


// to package shared parameters for an executor,
// we create a shared parameter matrix
// the rows correspond to levels of the executor's hierarchy
// the columns correspond to shared arguments
template<size_t... RowIndex, class... SharedArgs>
packaged_shared_parameters_t<tuple<SharedArgs...>>
  pack_shared_parameters_for_executor_impl(index_sequence<RowIndex...>,
                                           const tuple<SharedArgs...>& shared_arg_tuple)
{
  return detail::make_tuple(
    detail::tuple_map(detail::call_make<RowIndex>{}, shared_arg_tuple)...
  );
}


template<size_t num_rows, class... SharedArgs>
packaged_shared_parameters_t<tuple<SharedArgs...>>
  pack_shared_parameters_for_executor(const tuple<SharedArgs...>& shared_arg_tuple)
{
  return detail::pack_shared_parameters_for_executor_impl(
    make_index_sequence<num_rows>{},
    shared_arg_tuple
  );
}


template<class RowIndices, class TupleMatrix>
struct extracted_shared_parameters_t_impl;


template<size_t... RowIndices, class TupleMatrix>
struct extracted_shared_parameters_t_impl<index_sequence<RowIndices...>,TupleMatrix>
{
  using type = agency::detail::tuple<
    decltype(
      agency::detail::tuple_find_non_null(detail::get<RowIndices>(*std::declval<TupleMatrix*>()))
    )...
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
  return detail::tie(
    detail::tuple_find_non_null(detail::get<RowIndex>(mtx))...
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
unpack_shared_parameters_from_executor_t<tuple<Rows...>>
  unpack_shared_parameters_from_executor(tuple<Rows...>& shared_param_matrix)
{
  // to transform the shared_param_matrix into a tuple of shared parameters
  // we need to find the actual (non-null) parameter in each column of the matrix
  // the easiest way to do this is to tranpose the matrix and extract the shared parameter from each row of the transpose
  return detail::extract_shared_parameters_from_rows(
    detail::make_transposed_view<sizeof...(Rows)>(
      shared_param_matrix
    )
  );
}


template<size_t I, class Arg>
__AGENCY_ANNOTATION
typename std::enable_if<
  !is_shared_parameter<typename std::decay<Arg>::type>::value,
  Arg&&
>::type
  hold_shared_parameters_place(Arg&& arg)
{
  return std::forward<Arg>(arg);
}


template<size_t I, class T>
__AGENCY_ANNOTATION
typename std::enable_if<
  is_shared_parameter<typename std::decay<T>::type>::value,
  placeholder<I>
>::type
  hold_shared_parameters_place(T&&)
{
  return placeholder<I>{};
}


template<class... Args>
auto forward_shared_parameters_as_tuple(Args&&... args)
  -> decltype(
       __tu::tuple_filter_invoke<is_shared_parameter_ref>(
         detail::forward_as_tuple(std::forward<Args>(args)...),
         detail::forwarder()
       )
     )
{
  return __tu::tuple_filter_invoke<is_shared_parameter_ref>(
    detail::forward_as_tuple(std::forward<Args>(args)...),
    detail::forwarder()
  );
}


// if J... is a bit vector indicating which elements of args are shared parameters
// then I... is the exclusive scan of J
template<size_t... I, class Function, class... Args>
auto bind_unshared_parameters_impl(index_sequence<I...>, Function f, Args&&... args)
  -> decltype(
       std::bind(f, hold_shared_parameters_place<1 + I>(std::forward<Args>(args))...)
     )
{
  // we add 1 to I to account for the executor_idx argument which will be passed as the first parameter to f
  return std::bind(f, hold_shared_parameters_place<1 + I>(std::forward<Args>(args))...);
}


template<class... Args>
struct arg_is_shared
{
  using tuple_type = tuple<Args...>;

  template<size_t i>
  using map = is_shared_parameter<
    typename std::decay<
      typename std::tuple_element<i,tuple_type>::type
    >::type
  >;
};


// XXX nvcc 7.0 doesnt like agency::detail::scanned_shared_argument_indices
//     so WAR it by implementing a slightly different version here
template<class... Args>
struct scanned_shared_argument_indices_impl
{
  using type = agency::detail::transform_exclusive_scan_index_sequence<
    agency::detail::arg_is_shared<Args...>::template map,
    0,
    // XXX various compilers have difficulty with index_sequence_for, so WAR it
    //index_sequence_for<Args...>
    agency::detail::make_index_sequence<sizeof...(Args)>
  >;
};


template<class... Args>
using scanned_shared_argument_indices = typename scanned_shared_argument_indices_impl<Args...>::type;


template<class Function, class... Args>
auto bind_unshared_parameters(Function f, Args&&... args)
  -> decltype(
       bind_unshared_parameters_impl(scanned_shared_argument_indices<Args...>{}, f, std::forward<Args>(args)...)
     )
{
  return bind_unshared_parameters_impl(scanned_shared_argument_indices<Args...>{}, f, std::forward<Args>(args)...);
}


} // end detail
} // end agency

