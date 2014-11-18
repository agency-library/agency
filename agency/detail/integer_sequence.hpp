#pragma once

#include <type_traits>


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


} // end integer_sequence_detail


template<class _Tp, _Tp _Np>
using make_integer_sequence = typename integer_sequence_detail::make_integer_sequence<_Tp, _Np>::type;


template<size_t _Np>
using make_index_sequence = make_integer_sequence<size_t, _Np>;


template<class... _Tp>
using index_sequence_for = make_index_sequence<sizeof...(_Tp)>;
  

} // end detail
} // end agency

