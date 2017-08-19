#pragma once

#include <tuple>
#include <type_traits>
#include <agency/detail/config.hpp>
#include <agency/tuple.hpp>
#include <agency/detail/integer_sequence.hpp>

namespace agency
{
namespace detail
{


// a TupleMatrix is a tuple of tuples
// where each nested tuple is the same type
// the nested tuples are allowed to be values or references
// we can create views of its columns or a view of its transpose
// The degenerate case of an empty TupleMatrix with 0 rows and 0 columns
// is simply an empty tuple<>

template<class TupleMatrix, size_t num_rows = std::tuple_size<TupleMatrix>::value>
struct tuple_matrix_shape_impl
{
  static const size_t rows = num_rows;

  // decay any reference on the row type, if it exists
  using row_type = typename std::decay<
    typename std::tuple_element<0,TupleMatrix>::type
  >::type;

  static const size_t columns = std::tuple_size<row_type>::value;
};

template<class RowType>
struct tuple_matrix_shape_impl<RowType,0>
{
  static const size_t rows = 0;
  static const size_t cols = 0;
};

template<class TupleMatrix>
struct tuple_matrix_shape : tuple_matrix_shape_impl<TupleMatrix> {};


// returns the type of element at [row_idx, column_idx]
template<size_t row_idx, size_t column_idx, class Tuple>
struct tuple_matrix_element;


template<size_t row_idx, size_t column_idx, class... Rows>
struct tuple_matrix_element<row_idx,column_idx,tuple<Rows...>> :
  std::tuple_element<
    column_idx,
    typename std::decay<
      typename std::tuple_element<
        row_idx,
        tuple<Rows...>
      >::type
    >::type
  >
{};


// returns a reference to the element at [row_idx, column_idx]
template<size_t row_idx, size_t column_idx, class... Rows>
__AGENCY_ANNOTATION
typename tuple_matrix_element<row_idx, column_idx, tuple<Rows...>>::type &
  tuple_matrix_get(tuple<Rows...>& mtx)
{
  return agency::get<column_idx>(agency::get<row_idx>(mtx));
}


template<size_t column_idx, class RowIndices, class Tuple>
struct tuple_matrix_column_view_impl;


template<size_t column_idx, size_t... RowIndices, class Tuple>
struct tuple_matrix_column_view_impl<column_idx, index_sequence<RowIndices...>, Tuple>
{
  using type = tuple<
    typename tuple_matrix_element<RowIndices,column_idx,Tuple>::type&...
  >;
};


template<size_t column_idx, class Tuple>
using tuple_matrix_column_view = typename tuple_matrix_column_view_impl<
  column_idx,
  make_index_sequence<std::tuple_size<Tuple>::value>,
  Tuple
>::type;


template<size_t column_idx, class Indices>
struct make_tuple_matrix_column_view_impl_impl;


template<size_t column_idx, size_t... RowIndices>
struct make_tuple_matrix_column_view_impl_impl<column_idx, index_sequence<RowIndices...>>
{
  template<class... Rows>
  static __AGENCY_ANNOTATION
  tuple_matrix_column_view<column_idx, tuple<Rows...>> get_column(tuple<Rows...>& mtx)
  {
    return agency::tie(tuple_matrix_get<RowIndices,column_idx>(mtx)...);
  }
};


template<size_t column_idx, size_t num_rows, class... Rows>
__AGENCY_ANNOTATION
tuple_matrix_column_view<column_idx,tuple<Rows...>>
  make_tuple_matrix_column_view_impl(tuple<Rows...>& mtx)
{
  return make_tuple_matrix_column_view_impl_impl<column_idx, make_index_sequence<num_rows>>::get_column(mtx);
}


template<size_t column_idx, class... Rows>
__AGENCY_ANNOTATION
tuple_matrix_column_view<column_idx,tuple<Rows...>>
  make_tuple_matrix_column_view(tuple<Rows...>& mtx)
{
  return make_tuple_matrix_column_view_impl<column_idx,sizeof...(Rows)>(mtx);
}


template<class ColumnIndices, class Tuple>
struct tuple_matrix_transpose_view_impl;


template<size_t... ColumnIndices, class Tuple>
struct tuple_matrix_transpose_view_impl<index_sequence<ColumnIndices...>,Tuple>
{
  using type = tuple<
    tuple_matrix_column_view<ColumnIndices,Tuple>...
  >;
};


template<class Tuple>
using tuple_matrix_transpose_view = typename tuple_matrix_transpose_view_impl<
  make_index_sequence<
    tuple_matrix_shape<Tuple>::columns
  >,
  Tuple
>::type;


template<size_t... ColumnIndex, class... Rows>
__AGENCY_ANNOTATION
tuple_matrix_transpose_view<tuple<Rows...>>
  make_transposed_view_impl(index_sequence<ColumnIndex...>, tuple<Rows...>& mtx)
{
  // to create a transposed view of the matrix, create a tuple of the columns
  return agency::make_tuple(
    make_tuple_matrix_column_view<ColumnIndex>(mtx)...
  );
}


// XXX seems weird that this takes num_columns as a parameter
//     it should just be able to figure it out from tuple_matrix_shape
template<size_t num_columns, class... Rows>
__AGENCY_ANNOTATION
tuple_matrix_transpose_view<tuple<Rows...>>
  make_transposed_view(tuple<Rows...>& mtx)
{
  return detail::make_transposed_view_impl(
    make_index_sequence<num_columns>{},
    mtx
  );
}


} // end detail
} // end agency

