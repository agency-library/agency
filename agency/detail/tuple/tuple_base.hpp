// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/type_list.hpp>
#include <agency/detail/tuple/tuple_leaf.hpp>
#include <stddef.h> // XXX instead of <cstddef> to WAR clang issue
#include <type_traits>
#include <utility> // <utility> declares std::tuple_element et al. for us


namespace agency
{
namespace detail
{


// declare tuple_base so that the specializations in std:: below can refer to it
template<class IndexSequence, class... Args>
class tuple_base;


} // end detail
} // end agency


// specializations of stuff in std come before their use below in the definition of tuple_base
namespace std
{


template<size_t i, class IndexSequence>
class tuple_element<i, agency::detail::tuple_base<IndexSequence>> {};


template<class IndexSequence, class Type1, class... Types>
class tuple_element<0, agency::detail::tuple_base<IndexSequence,Type1,Types...>>
{
  public:
    using type = Type1;
};


template<size_t i, class IndexSequence, class Type1, class... Types>
class tuple_element<i, agency::detail::tuple_base<IndexSequence,Type1,Types...>>
{
  public:
    using type = typename tuple_element<i - 1, agency::detail::tuple_base<IndexSequence,Types...>>::type;
};


template<class IndexSequence, class... Types>
class tuple_size<agency::detail::tuple_base<IndexSequence,Types...>>
  : public std::integral_constant<size_t, sizeof...(Types)>
{};


} // end std


namespace agency
{
namespace detail
{


// XXX this implementation is based on Howard Hinnant's "tuple leaf" construction in libcxx
template<size_t... I, class... Types>
class tuple_base<index_sequence<I...>, Types...>
  : public tuple_leaf<I,Types>...
{
  public:
    tuple_base() = default;


    __AGENCY_ANNOTATION
    tuple_base(const Types&... args)
      : tuple_leaf<I,Types>(args)...
    {}


    template<class... UTypes,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == sizeof...(UTypes)) &&
               conjunction<
                 std::is_constructible<Types,UTypes&&>...
               >::value
             )>
    __AGENCY_ANNOTATION
    explicit tuple_base(UTypes&&... args)
      : tuple_leaf<I,Types>(std::forward<UTypes>(args))...
    {}


    __AGENCY_ANNOTATION
    tuple_base(const tuple_base& other)
      : tuple_leaf<I,Types>(other.template const_leaf<I>())...
    {}


    __AGENCY_ANNOTATION
    tuple_base(tuple_base&& other)
      : tuple_leaf<I,Types>(std::move(other.template mutable_leaf<I>()))...
    {}


    template<class... UTypes,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == sizeof...(UTypes)) &&
               conjunction<
                 std::is_constructible<Types,const UTypes&>...
                >::value
             )>
    __AGENCY_ANNOTATION
    tuple_base(const tuple_base<index_sequence<I...>,UTypes...>& other)
      : tuple_leaf<I,Types>(other.template const_leaf<I>())...
    {}


    template<class... UTypes,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == sizeof...(UTypes)) &&
               conjunction<
                 std::is_constructible<Types,UTypes&&>...
                >::value
             )>
    __AGENCY_ANNOTATION
    tuple_base(tuple_base<index_sequence<I...>,UTypes...>&& other)
      : tuple_leaf<I,Types>(std::move(other.template mutable_leaf<I>()))...
    {}


    template<class... UTypes,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == sizeof...(UTypes)) &&
               conjunction<
                 std::is_constructible<Types,const UTypes&>...
                >::value
             )>
    __AGENCY_ANNOTATION
    tuple_base(const std::tuple<UTypes...>& other)
      : tuple_base{std::get<I>(other)...}
    {}


    __AGENCY_ANNOTATION
    tuple_base& operator=(const tuple_base& other)
    {
      swallow((mutable_leaf<I>() = other.template const_leaf<I>())...);
      return *this;
    }


    __AGENCY_ANNOTATION
    tuple_base& operator=(tuple_base&& other)
    {
      swallow((mutable_leaf<I>() = std::move(other.template mutable_leaf<I>()))...);
      return *this;
    }


    template<class... UTypes,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == sizeof...(UTypes)) &&
               conjunction<
                 std::is_assignable<Types,const UTypes&>...
                >::value
             )>
    __AGENCY_ANNOTATION
    tuple_base& operator=(const tuple_base<index_sequence<I...>,UTypes...>& other)
    {
      swallow((mutable_leaf<I>() = other.template const_leaf<I>())...);
      return *this;
    }


    template<class... UTypes,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == sizeof...(UTypes)) &&
               conjunction<
                 std::is_assignable<Types,UTypes&&>...
               >::value
             )>
    __AGENCY_ANNOTATION
    tuple_base& operator=(tuple_base<index_sequence<I...>,UTypes...>&& other)
    {
      swallow((mutable_leaf<I>() = std::move(other.template mutable_leaf<I>()))...);
      return *this;
    }


    template<class UType1, class UType2,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == 2) &&
               conjunction<
                 std::is_assignable<typename std::tuple_element<                            0,tuple_base>::type,const UType1&>,
                 std::is_assignable<typename std::tuple_element<sizeof...(Types) == 2 ? 1 : 0,tuple_base>::type,const UType2&>
               >::value
             )>
    __AGENCY_ANNOTATION
    tuple_base& operator=(const std::pair<UType1,UType2>& p)
    {
      mutable_get<0>() = p.first;
      mutable_get<1>() = p.second;
      return *this;
    }


    template<class UType1, class UType2,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == 2) &&
               conjunction<
                 std::is_assignable<typename std::tuple_element<                            0,tuple_base>::type,UType1&&>,
                 std::is_assignable<typename std::tuple_element<sizeof...(Types) == 2 ? 1 : 0,tuple_base>::type,UType2&&>
               >::value
             )>
    __AGENCY_ANNOTATION
    tuple_base& operator=(std::pair<UType1,UType2>&& p)
    {
      mutable_get<0>() = std::move(p.first);
      mutable_get<1>() = std::move(p.second);
      return *this;
    }


    template<size_t i>
    __AGENCY_ANNOTATION
    const tuple_leaf<i,typename std::tuple_element<i,tuple_base>::type>& const_leaf() const
    {
      return *this;
    }


    template<size_t i>
    __AGENCY_ANNOTATION
    tuple_leaf<i,typename std::tuple_element<i,tuple_base>::type>& mutable_leaf()
    {
      return *this;
    }


    template<size_t i>
    __AGENCY_ANNOTATION
    tuple_leaf<i,typename std::tuple_element<i,tuple_base>::type>&& move_leaf() &&
    {
      return std::move(*this);
    }


    __AGENCY_ANNOTATION
    void swap(tuple_base& other)
    {
      swallow(tuple_leaf<I,Types>::swap(other)...);
    }


    template<size_t i>
    __AGENCY_ANNOTATION
    const typename std::tuple_element<i,tuple_base>::type& const_get() const
    {
      return const_leaf<i>().const_get();
    }


    template<size_t i>
    __AGENCY_ANNOTATION
    typename std::tuple_element<i,tuple_base>::type& mutable_get()
    {
      return mutable_leaf<i>().mutable_get();
    }


    // enable conversion to Tuple-like things
    // XXX relax std::tuple to Tuple and require is_tuple
    template<class... UTypes,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == sizeof...(UTypes)) &&
               conjunction<
                 std::is_constructible<Types,const UTypes&>...
                >::value
             )>
    __AGENCY_ANNOTATION
    operator std::tuple<UTypes...> () const
    {
      return std::tuple<UTypes...>(const_get<I>()...);
    }


  private:
    template<class... Args>
    __AGENCY_ANNOTATION
    static void swallow(Args&&...) {}
};


} // end detail
} // end agency

