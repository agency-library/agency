#pragma once

#include <type_traits>
#include <stddef.h> // for size_t


namespace agency
{
namespace detail
{


template<class _Tp, _Tp... _Ip>
struct integer_sequence
{
  typedef _Tp value_type;
  static_assert(std::is_integral<_Tp>::value,
                "std::integer_sequence can only be instantiated with an integral type" );
  static constexpr size_t size() noexcept { return sizeof...(_Ip); }
};


template<size_t... _Ip>
using index_sequence = integer_sequence<size_t, _Ip...>;


namespace integer_sequence_detail
{

template <class _Tp, _Tp _Sp, _Tp _Ep, class _IntSequence>
struct make_integer_sequence_unchecked;

template <class _Tp, _Tp _Sp, _Tp _Ep, _Tp ..._Indices>
struct make_integer_sequence_unchecked<_Tp, _Sp, _Ep,
                                       integer_sequence<_Tp, _Indices...>>
{
  typedef typename make_integer_sequence_unchecked<
    _Tp, _Sp+1, _Ep,
    integer_sequence<_Tp, _Indices..., _Sp>
  >::type type;
};


template <class _Tp, _Tp _Ep, _Tp ..._Indices>
struct make_integer_sequence_unchecked<_Tp, _Ep, _Ep,
                                       integer_sequence<_Tp, _Indices...>>
{
  typedef integer_sequence<_Tp, _Indices...> type;
};


template <class _Tp, _Tp _Ep>
struct make_integer_sequence
{
  static_assert(std::is_integral<_Tp>::value,
                "std::make_integer_sequence can only be instantiated with an integral type" );
  static_assert(0 <= _Ep, "std::make_integer_sequence input shall not be negative");
  typedef typename make_integer_sequence_unchecked
                   <
                      _Tp, 0, _Ep, integer_sequence<_Tp>
                   >::type type;
};



template<template<size_t> class MetaFunction, class IndexSeqence>
struct map_index_sequence;


template<template<size_t> class MetaFunction, size_t... Indices>
struct map_index_sequence<MetaFunction, index_sequence<Indices...>>
{
  using type = index_sequence<MetaFunction<Indices>::value...>;
};


template<class IndexSequence1, class IndexSequence2>
struct index_sequence_cat_impl;

template<size_t... Indices1, size_t... Indices2>
struct index_sequence_cat_impl<index_sequence<Indices1...>,index_sequence<Indices2...>>
{
  using type = index_sequence<Indices1...,Indices2...>;
};

template<class IndexSequence1, class IndexSequence2>
using index_sequence_cat = typename index_sequence_cat_impl<IndexSequence1,IndexSequence2>::type;

// compute the exclusive scan of IndexSequence
// initializing the first value in the sequence to Init
template<size_t Init, class IndexSequence>
struct exclusive_scan_index_sequence;

template<size_t Init, size_t Index0, size_t... Indices>
struct exclusive_scan_index_sequence<Init,index_sequence<Index0, Indices...>>
{
  using rest = typename exclusive_scan_index_sequence<Init + Index0, index_sequence<Indices...>>::type; 

  using type = index_sequence_cat<index_sequence<Init>, rest>;
};

template<size_t Init, size_t Index0>
struct exclusive_scan_index_sequence<Init,index_sequence<Index0>>
{
  using type = index_sequence<Init>;
};


} // end integer_sequence_detail


template<class _Tp, _Tp _Np>
using make_integer_sequence = typename integer_sequence_detail::make_integer_sequence<_Tp, _Np>::type;


template<size_t _Np>
using make_index_sequence = make_integer_sequence<size_t, _Np>;


template<class...>
struct sizeof_parameter_pack;

template<>
struct sizeof_parameter_pack<> : std::integral_constant<size_t,0> {};

template<class T, class... Types>
struct sizeof_parameter_pack<T,Types...>
  : std::integral_constant<
      size_t,
      1 + sizeof_parameter_pack<Types...>::value
    >
{};


// XXX workaround nvbug 1668372
//template<class... _Tp>
//using index_sequence_for = make_index_sequence<sizeof...(_Tp)>;
template<class... _Tp>
struct index_sequence_for_workaround_nvbug1668372
{
  // XXX workaround nvbug 1668718
  //using type = make_index_sequence<sizeof...(_Tp)>;
  using type = make_index_sequence<sizeof_parameter_pack<_Tp...>::value>;
};

template<class... _Tp>
using index_sequence_for = typename index_sequence_for_workaround_nvbug1668372<_Tp...>::type;


template<template<size_t> class MetaFunction, class IndexSequence>
using map_index_sequence = typename integer_sequence_detail::map_index_sequence<MetaFunction,IndexSequence>::type;


template<size_t Init, class IndexSequence>
using exclusive_scan_index_sequence = typename integer_sequence_detail::exclusive_scan_index_sequence<Init,IndexSequence>::type;


template<template<size_t> class MetaFunction, size_t Init, class IndexSequence>
using transform_exclusive_scan_index_sequence = exclusive_scan_index_sequence<Init, map_index_sequence<MetaFunction,IndexSequence>>;


template<size_t i, class IndexSequence>
struct index_sequence_element;


template<size_t Index0, size_t... Indices>
struct index_sequence_element<0, index_sequence<Index0, Indices...>>
  : std::integral_constant<
      size_t,
      Index0
    >
{};


template<size_t i, size_t Index0, size_t... Indices>
struct index_sequence_element<i,index_sequence<Index0,Indices...>>
  : index_sequence_element<
      i-1,
      index_sequence<Indices...>
    >
{};


template<class IndexSequence>
struct index_sequence_size;

template<size_t... Indices>
struct index_sequence_size<index_sequence<Indices...>>
  : std::integral_constant<
      size_t,
      sizeof...(Indices)
    >
{};


namespace integer_sequence_detail
{



template<class IndexSequence, class IndexSequenceMap, class IndexSequenceInput>
struct index_sequence_gather_impl;

template<size_t... Indices, class IndexSequenceMap, class IndexSequenceInput>
struct index_sequence_gather_impl<index_sequence<Indices...>, IndexSequenceMap, IndexSequenceInput>
{
  using type = index_sequence<
    index_sequence_element<
      index_sequence_element<
        Indices,
        IndexSequenceMap
      >::value,
      IndexSequenceInput
    >::value...
  >;
};


} // end integer_sequence_detail


template<class IndexSequenceMap, class IndexSequenceInput>
using index_sequence_gather = typename integer_sequence_detail::index_sequence_gather_impl<
  make_index_sequence<index_sequence_size<IndexSequenceMap>::value>,
  IndexSequenceMap,
  IndexSequenceInput
>::type;
  

} // end detail
} // end agency

