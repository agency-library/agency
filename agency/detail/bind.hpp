#pragma once

#include <functional>
#include <type_traits>

namespace agency
{
namespace detail
{


using std::bind;


// we define our own placeholders that NVCC can consume
template<int I>
struct placeholder {};


namespace placeholders
{


constexpr placeholder<0>   _1{};
constexpr placeholder<1>   _2{};
constexpr placeholder<2>   _3{};
constexpr placeholder<3>   _4{};
constexpr placeholder<4>   _5{};
constexpr placeholder<5>   _6{};
constexpr placeholder<6>   _7{};
constexpr placeholder<7>   _8{};
constexpr placeholder<8>   _9{};
constexpr placeholder<9>   _10{};


} // end placeholders
} // end detail
} // end agency


namespace std
{


template<int I>
struct is_placeholder<agency::detail::placeholder<I>> : std::integral_constant<int,I+1> {};


} // end std

