#pragma once

#include <agency/detail/tuple_utility.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/tuple.hpp>
#include <type_traits>
#include <utility>
#include <functional>

namespace agency
{
namespace detail
{


template<class Tuple,
         class Enable = typename std::enable_if<
           (std::tuple_size<typename std::decay<Tuple>::type>::value == 0)
         >::type
        >
void tuple_find_if_impl(Tuple&& filtered_tuple) {}


template<class Tuple,
         class Enable = typename std::enable_if<
           (std::tuple_size<typename std::decay<Tuple>::type>::value != 0)
         >::type
        >
auto tuple_find_if_impl(Tuple&& filtered_tuple)
  -> decltype(
       detail::get<0>(std::forward<Tuple>(filtered_tuple))
     )
{
  return detail::get<0>(std::forward<Tuple>(filtered_tuple));
}


template<template<class> class MetaFunction, class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_find_if(Tuple&& t)
  -> decltype(
       tuple_find_if_impl(__tu::tuple_filter_invoke<MetaFunction>(std::forward<Tuple>(t), forwarder{}))
     )
{
  return tuple_find_if_impl(__tu::tuple_filter_invoke<MetaFunction>(std::forward<Tuple>(t), forwarder{}));
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


template<class Tuple>
auto tuple_find_non_null(Tuple&& t)
  -> decltype(tuple_find_if<is_not_null>(std::forward<Tuple>(t)))
{
  return tuple_find_if<is_not_null>(std::forward<Tuple>(t));
}


template<size_t level, class T, class... Args>
struct shared_parameter
{
  T make(std::integral_constant<size_t,level>) const
  {
    return __tu::make_from_tuple<T>(args_);
  }

  template<size_t other_level>
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
  auto operator()(const shared_parameter<other_level,T,Args...>& parm) const
    -> decltype(
         parm.make(std::integral_constant<size_t,level>{})
       )
  {
    return parm.make(std::integral_constant<size_t,level>{});
  }
};


template<size_t column_idx, size_t... row_idx, class TupleMatrix>
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


template<size_t... RowIndex, class SharedArgTuple>
auto pack_shared_parameters_for_executor_impl(index_sequence<RowIndex...>,
                                              SharedArgTuple&& shared_arg_tuple)
  -> decltype(
       detail::make_tuple(
         __tu::tuple_map(call_make<RowIndex>{}, std::forward<SharedArgTuple>(shared_arg_tuple))...
       )
     )
{
  return detail::make_tuple(
    __tu::tuple_map(call_make<RowIndex>{}, std::forward<SharedArgTuple>(shared_arg_tuple))...
  );
}


// to package shared parameters for an executor,
// we create a shared parameter matrix
// the rows correspond to levels of the executor's hierarchy
// the columns correspond to shared arguments
template<size_t num_rows, class SharedArgTuple>
auto pack_shared_parameters_for_executor(SharedArgTuple&& shared_arg_tuple)
  -> decltype(
       pack_shared_parameters_for_executor_impl(
         make_index_sequence<num_rows>{},
         std::forward<SharedArgTuple>(shared_arg_tuple)
       )
     )
{
  return pack_shared_parameters_for_executor_impl(
    make_index_sequence<num_rows>{},
    std::forward<SharedArgTuple>(shared_arg_tuple)
  );
}


template<size_t... RowIndex, class TupleMatrix>
auto extract_shared_parameters_from_rows_impl(index_sequence<RowIndex...>, TupleMatrix&& mtx)
  -> decltype(
       detail::tie(
         tuple_find_non_null(detail::get<RowIndex>(std::forward<TupleMatrix>(mtx)))...
       )
     )
{
  return detail::tie(
    tuple_find_non_null(detail::get<RowIndex>(mtx))...
  );
}


template<class TupleMatrix>
auto extract_shared_parameters_from_rows(TupleMatrix&& mtx)
  -> decltype(
       extract_shared_parameters_from_rows_impl(
         make_index_sequence<
           std::tuple_size<
             typename std::decay<TupleMatrix>::type
           >::value
         >{},
         std::forward<TupleMatrix>(mtx)
       )
     )
{
  return extract_shared_parameters_from_rows_impl(
    make_index_sequence<
      std::tuple_size<
        typename std::decay<TupleMatrix>::type
      >::value
    >{},
    std::forward<TupleMatrix>(mtx)
  );
}


template<size_t... ColumnIndex, class TupleMatrix>
auto make_transposed_view_impl(index_sequence<ColumnIndex...>, TupleMatrix&& mtx)
  -> decltype(
       detail::make_tuple(
         tuple_matrix_column<ColumnIndex>(std::forward<TupleMatrix>(mtx))...
       )
     )
{
  // to create a transposed view of the matrix, create a tuple of the columns
  return detail::make_tuple(
    tuple_matrix_column<ColumnIndex>(std::forward<TupleMatrix>(mtx))...
  );
}


template<size_t num_columns, class TupleMatrix>
auto make_transposed_view(TupleMatrix&& mtx)
  -> decltype(
       make_transposed_view_impl(
         make_index_sequence<num_columns>{},
         std::forward<TupleMatrix>(mtx)
       )
     )
{
  return make_transposed_view_impl(
    make_index_sequence<num_columns>{},
    std::forward<TupleMatrix>(mtx)
  );
}


// a TupleMatrix is a tuple of tuples
// where each nested tuple is the same type
template<class TupleMatrix>
struct tuple_matrix_shape
{
  static const size_t rows = std::tuple_size<TupleMatrix>::value;

  using row_type = typename std::decay<
    typename std::tuple_element<0,TupleMatrix>::type
  >::type;

  static const size_t columns = std::tuple_size<row_type>::value;
};


// the number of shared parameters is equal to the number of columns of TupleMatrix
// this is equal to the tuple_size of tuple_element 0
template<class TupleMatrix,
         size_t NumSharedParams = tuple_matrix_shape<typename std::decay<TupleMatrix>::type>::columns
        >
auto unpack_shared_parameters_from_executor(TupleMatrix&& shared_param_matrix)
  -> decltype(
       extract_shared_parameters_from_rows(
         make_transposed_view<NumSharedParams>(
           std::forward<TupleMatrix>(shared_param_matrix)
         )
       )
     )
{
  // to transform the shared_param_matrix into a tuple of shared parameters
  // we need to find the actual (non-null) parameter in each column of the matrix
  // the easiest way to do this is to tranpose the matrix and extract the shared parameter from each row of the transpose
  return extract_shared_parameters_from_rows(
    make_transposed_view<NumSharedParams>(
      std::forward<TupleMatrix>(shared_param_matrix)
    )
  );
}


template<size_t I, class Arg>
typename std::enable_if<
  !is_shared_parameter<typename std::decay<Arg>::type>::value,
  Arg&&
>::type
  hold_shared_parameters_place(Arg&& arg)
{
  return std::forward<Arg>(arg);
}


using placeholder_tuple_t = tuple<
  decltype(std::placeholders::_1),
  decltype(std::placeholders::_2),
  decltype(std::placeholders::_3),
  decltype(std::placeholders::_4),
  decltype(std::placeholders::_5),
  decltype(std::placeholders::_6),
  decltype(std::placeholders::_7),
  decltype(std::placeholders::_8),
  decltype(std::placeholders::_9),
  decltype(std::placeholders::_10)
>;


template<size_t i>
using placeholder = typename std::tuple_element<i, placeholder_tuple_t>::type;


template<size_t I, class T>
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


// we need to generate
template<class... Args>
using scanned_shared_argument_indices = transform_exclusive_scan_index_sequence<
  arg_is_shared<Args...>::template map,
  0,
  index_sequence_for<Args...>
>;


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

