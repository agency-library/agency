#pragma once

#include <agency/detail/tuple_utility.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/tuple_matrix.hpp>
#include <agency/detail/bind.hpp>
#include <agency/functional.hpp>
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
  return detail::get<find_exactly_one_not_null<Types...>::value>(t);
}


template<size_t level>
struct make_shared_parameter_matrix_element
{
  // XXX might want to std::move parm
  template<class T, class... Args>
  __AGENCY_ANNOTATION
  shared_parameter<level,T,Args...> operator()(const shared_parameter<level,T,Args...>& parm) const
  {
    return parm;
  }

  template<size_t other_level, class T, class... Args>
  __AGENCY_ANNOTATION
  null_type operator()(const shared_parameter<other_level,T,Args...>&) const
  {
    return null_type{};
  }
};


template<class Indices, class Tuple>
struct shared_parameter_matrix_t_impl;


template<size_t... RowIndex, class Tuple>
struct shared_parameter_matrix_t_impl<index_sequence<RowIndex...>,Tuple>
{
  using type = decltype(
    detail::make_tuple(
      detail::tuple_map(make_shared_parameter_matrix_element<RowIndex>{}, std::declval<Tuple>())...
    )
  );
};


template<size_t num_rows, class Tuple>
using shared_parameter_matrix_t = typename shared_parameter_matrix_t_impl<
  agency::detail::make_index_sequence<
    num_rows
  >,
  Tuple
>::type;


// to package shared parameters for an executor,
// we create a shared parameter matrix
// the rows correspond to levels of the executor's hierarchy
// the columns correspond to shared arguments
template<size_t... RowIndex, class... SharedArgs>
shared_parameter_matrix_t<sizeof...(RowIndex), tuple<SharedArgs...>>
  make_shared_parameter_matrix_impl(index_sequence<RowIndex...>,
                                    const tuple<SharedArgs...>& shared_arg_tuple)
{
  return detail::make_tuple(
    detail::tuple_map(make_shared_parameter_matrix_element<RowIndex>{}, shared_arg_tuple)...
  );
}


template<size_t execution_depth, class... SharedArgs>
shared_parameter_matrix_t<execution_depth, tuple<SharedArgs...>>
  make_shared_parameter_matrix(const tuple<SharedArgs...>& shared_arg_tuple)
{
  return detail::make_shared_parameter_matrix_impl(
    make_index_sequence<execution_depth>{},
    shared_arg_tuple
  );
}


template<size_t executor_depth, class... SharedArgs>
shared_parameter_matrix_t<executor_depth, tuple<SharedArgs...>>
  make_shared_parameter_package_for_executor(const tuple<SharedArgs...>& shared_arg_tuple)
{
  return make_shared_parameter_matrix<executor_depth>(shared_arg_tuple);
}


template<class RowIndices, class TupleMatrix>
struct extracted_shared_parameters_t_impl;


template<size_t... RowIndices, class TupleMatrix>
struct extracted_shared_parameters_t_impl<index_sequence<RowIndices...>,TupleMatrix>
{
  using type = agency::detail::tuple<
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
  return detail::unpack_shared_parameter_matrix_from_executor(detail::tie(shared_params...));
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

