// Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
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

#include <stddef.h> // XXX instead of <cstddef> to WAR clang issue
#include <type_traits>
#include <utility>

// allow the user to define an annotation to apply to these functions
// by default, it attempts to be constexpr
#ifndef __TUPLE_ANNOTATION
#  if __cplusplus <= 201103L
#    define __TUPLE_ANNOTATION
#  else
#    define __TUPLE_ANNOTATION constexpr
#  endif
#  define __TUPLE_ANNOTATION_NEEDS_UNDEF
#endif

// allow the user to define a namespace for these functions
#ifndef __TUPLE_NAMESPACE
#define __TUPLE_NAMESPACE std
#define __TUPLE_NAMESPACE_NEEDS_UNDEF
#endif


namespace __TUPLE_NAMESPACE
{

template<class... Types> class tuple;

} // end namespace


// specializations of stuff in std come before their use
namespace std
{


template<size_t, class> struct tuple_element;


template<size_t i>
struct tuple_element<i, __TUPLE_NAMESPACE::tuple<>> {};


template<class Type1, class... Types>
struct tuple_element<0, __TUPLE_NAMESPACE::tuple<Type1,Types...>>
{
  using type = Type1;
};


template<size_t i, class Type1, class... Types>
struct tuple_element<i, __TUPLE_NAMESPACE::tuple<Type1,Types...>>
{
  using type = typename tuple_element<i - 1, __TUPLE_NAMESPACE::tuple<Types...>>::type;
};


template<size_t i, class... Types>
using tuple_element_t = typename tuple_element<i,Types...>::type;


template<class> struct tuple_size;


template<class... Types>
struct tuple_size<__TUPLE_NAMESPACE::tuple<Types...>>
  : std::integral_constant<size_t, sizeof...(Types)>
{};


} // end std


namespace __TUPLE_NAMESPACE
{

// define variadic "and" operator 
// prefix with "__tuple" to avoid collisions with other implementations 
template <typename... Conditions>
  struct __tuple_and;

template<>
  struct __tuple_and<>
    : public std::true_type
{
};

template <typename Condition, typename... Conditions>
  struct __tuple_and<Condition, Conditions...>
    : public std::integral_constant<
        bool,
        Condition::value && __tuple_and<Conditions...>::value>
{
};

// XXX this implementation is based on Howard Hinnant's "tuple leaf" construction in libcxx


// define index sequence in case it is missing
// prefix this stuff with "tuple" to avoid collisions with other implementations
template<size_t... I> struct __tuple_index_sequence {};

template<size_t Start, typename Indices, size_t End>
struct __tuple_make_index_sequence_impl;

template<size_t Start, size_t... Indices, size_t End>
struct __tuple_make_index_sequence_impl<
  Start,
  __tuple_index_sequence<Indices...>, 
  End
>
{
  typedef typename __tuple_make_index_sequence_impl<
    Start + 1,
    __tuple_index_sequence<Indices..., Start>,
    End
  >::type type;
};

template<size_t End, size_t... Indices>
struct __tuple_make_index_sequence_impl<End, __tuple_index_sequence<Indices...>, End>
{
  typedef __tuple_index_sequence<Indices...> type;
};

template<size_t N>
using __tuple_make_index_sequence = typename __tuple_make_index_sequence_impl<0, __tuple_index_sequence<>, N>::type;

template<size_t I, class T>
class __tuple_leaf
{
  public:
    __TUPLE_ANNOTATION
    __tuple_leaf() = default;

    template<class U,
             class = typename std::enable_if<
               std::is_constructible<T,U>::value
             >::type>
    __TUPLE_ANNOTATION
    __tuple_leaf(U&& arg) : val_(std::forward<U>(arg)) {}

    template<class U,
             class = typename std::enable_if<
               std::is_constructible<T,U>::value
             >::type>
    __TUPLE_ANNOTATION
    __tuple_leaf(const __tuple_leaf<I,U>& other) : val_(other.const_get()) {}


    template<class U,
             class = typename std::enable_if<
               std::is_assignable<T,U>::value
             >::type>
    __TUPLE_ANNOTATION
    __tuple_leaf& operator=(const __tuple_leaf<I,U>& other)
    {
      mutable_get() = other.const_get();
      return *this;
    }

    template<class U,
             class = typename std::enable_if<
               std::is_assignable<T,U&&>::value
             >::type>
    __TUPLE_ANNOTATION
    __tuple_leaf& operator=(__tuple_leaf<I,U>&& other)
    {
      mutable_get() = std::move(other.mutable_get());
      return *this;
    }

    __TUPLE_ANNOTATION
    const T& const_get() const
    {
      return val_;
    }
  
    __TUPLE_ANNOTATION
    T& mutable_get()
    {
      return val_;
    }

    __TUPLE_ANNOTATION
    int swap(__tuple_leaf& other)
    {
      using std::swap;
      swap(mutable_get(), other.mutable_get());
      return 0;
    }

  private:
    T val_; // XXX apply empty base class optimization to this
};

template<class... Args>
struct __type_list {};

template<size_t i, class... Args>
struct __type_at_impl;

template<size_t i, class Arg0, class... Args>
struct __type_at_impl<i, Arg0, Args...>
{
  using type = typename __type_at_impl<i-1, Args...>::type;
};

template<class Arg0, class... Args>
struct __type_at_impl<0, Arg0,Args...>
{
  using type = Arg0;
};

template<size_t i, class... Args>
using __type_at = typename __type_at_impl<i,Args...>::type;

template<class IndexSequence, class... Args>
class __tuple_base;

template<size_t... I, class... Types>
class __tuple_base<__tuple_index_sequence<I...>, Types...>
  : public __tuple_leaf<I,Types>...
{
  public:
    using leaf_types = __type_list<__tuple_leaf<I,Types>...>;

    __TUPLE_ANNOTATION
    __tuple_base() = default;

    __TUPLE_ANNOTATION
    __tuple_base(const Types&... args)
      : __tuple_leaf<I,Types>(args)...
    {}


    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               __tuple_and<
                 std::is_constructible<Types,UTypes&&>...
               >::value
             >::type>
    __TUPLE_ANNOTATION
    explicit __tuple_base(UTypes&&... args)
      : __tuple_leaf<I,Types>(std::forward<UTypes>(args))...
    {}


    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               __tuple_and<
                 std::is_constructible<Types,const UTypes&>...
                >::value
             >::type>
    __TUPLE_ANNOTATION
    __tuple_base(const __tuple_base<__tuple_index_sequence<I...>,UTypes...>& other)
      : __tuple_leaf<I,Types>(other)...
    {}

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               __tuple_and<
                 std::is_assignable<Types,const UTypes&>...
                >::value
             >::type>
    __TUPLE_ANNOTATION
    __tuple_base& operator=(const __tuple_base<__tuple_index_sequence<I...>,UTypes...>& other)
    {
      swallow((mutable_leaf<I>() = other.template const_leaf<I>())...);
      return *this;
    }

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               __tuple_and<
                 std::is_assignable<Types,UTypes&&>...
               >::value
             >::type>
    __TUPLE_ANNOTATION
    __tuple_base& operator=(__tuple_base<__tuple_index_sequence<I...>,UTypes...>&& other)
    {
      swallow((mutable_leaf<I>() = std::move(other.template mutable_leaf<I>()))...);
      return *this;
    }

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               __tuple_and<
                 std::is_constructible<Types,const UTypes&>...
                >::value
             >::type>
    __TUPLE_ANNOTATION
    __tuple_base(const std::tuple<UTypes...>& other)
      : __tuple_base{std::get<I>(other)...}
    {}

    template<size_t i>
    __TUPLE_ANNOTATION
    const __tuple_leaf<i,__type_at<i,Types...>>& const_leaf() const
    {
      return *this;
    }

    template<size_t i>
    __TUPLE_ANNOTATION
    __tuple_leaf<i,__type_at<i,Types...>>& mutable_leaf()
    {
      return *this;
    }

    template<size_t i>
    __TUPLE_ANNOTATION
    __tuple_leaf<i,__type_at<i,Types...>>&& move_leaf() &&
    {
      return std::move(*this);
    }

    __TUPLE_ANNOTATION
    void swap(__tuple_base& other)
    {
      swallow(__tuple_leaf<I,Types>::swap(other)...);
    }

    // enable conversion to Tuple-like things
    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               __tuple_and<
                 std::is_constructible<Types,const UTypes&>...
                >::value
             >::type>
    __TUPLE_ANNOTATION
    operator std::tuple<UTypes...> () const
    {
      return std::tuple<UTypes...>(const_get<I>()...);
    }

    template<size_t i>
    __TUPLE_ANNOTATION
    const __type_at<i,Types...>& const_get() const
    {
      return const_leaf<i>().const_get();
    }

    template<size_t i>
    __TUPLE_ANNOTATION
    __type_at<i,Types...>& mutable_get()
    {
      return mutable_leaf<i>().mutable_get();
    }

  private:
    template<class... Args>
    __TUPLE_ANNOTATION
    static void swallow(Args&&...) {}
};


template<class... Types>
class tuple
{
  public:
    __TUPLE_ANNOTATION
    tuple() : base_{} {};

    __TUPLE_ANNOTATION
    explicit tuple(const Types&... args)
      : base_{args...}
    {}

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               __tuple_and<
                 std::is_constructible<Types,UTypes&&>...
               >::value
             >::type>
    __TUPLE_ANNOTATION
    explicit tuple(UTypes&&... args)
      : base_{std::forward<UTypes>(args)...}
    {}

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
                 __tuple_and<
                   std::is_constructible<Types,const UTypes&>...
                 >::value
             >::type>
    __TUPLE_ANNOTATION
    tuple(const tuple<UTypes...>& other)
      : base_{other.base_}
    {}

    template<class... UTypes>
    __TUPLE_ANNOTATION
    tuple(tuple<UTypes...>&& other)
      : base_{std::move(other.base_)}
    {}

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               __tuple_and<
                 std::is_constructible<__type_at<                            0,Types...>,const UType1&>,
                 std::is_constructible<__type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,const UType2&>
               >::value
             >::type>
    __TUPLE_ANNOTATION
    tuple(const std::pair<UType1,UType2>& p)
      : base_{p.first, p.second}
    {}

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               __tuple_and<
                 std::is_constructible<__type_at<                            0,Types...>,UType1&&>,
                 std::is_constructible<__type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,UType2&&>
               >::value
             >::type>
    __TUPLE_ANNOTATION
    tuple(std::pair<UType1,UType2>&& p)
      : base_{std::move(p.first), std::move(p.second)}
    {}

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
                 __tuple_and<
                   std::is_constructible<Types,const UTypes&>...
                 >::value
             >::type>
    __TUPLE_ANNOTATION
    tuple(const std::tuple<UTypes...>& other)
      : base_{other}
    {}

    __TUPLE_ANNOTATION
    tuple(const tuple& other)
      : base_{other.base_}
    {}

    __TUPLE_ANNOTATION
    tuple(tuple&& other)
      : base_{std::move(other.base_)}
    {}

    __TUPLE_ANNOTATION
    tuple& operator=(const tuple& other)
    {
      base_.operator=(other.base_);
      return *this;
    }

    __TUPLE_ANNOTATION
    tuple& operator=(tuple&& other)
    {
      base_.operator=(std::move(other.base_));
      return *this;
    }

    template<class... UTypes>
    __TUPLE_ANNOTATION
    tuple& operator=(const tuple<UTypes...>& other)
    {
      base_.operator=(other.base_);
      return *this;
    }

    template<class... UTypes>
    __TUPLE_ANNOTATION
    tuple& operator=(tuple<UTypes...>&& other)
    {
      base_.operator=(other.base_);
      return *this;
    }

    // XXX TODO assign from pair

    __TUPLE_ANNOTATION
    void swap(tuple& other)
    {
      base_.swap(other.base_);
    }

    // enable conversion to Tuple-like things
    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               __tuple_and<
                 std::is_constructible<Types,const UTypes&>...
                >::value
             >::type>
    __TUPLE_ANNOTATION
    operator std::tuple<UTypes...> () const
    {
      return static_cast<std::tuple<UTypes...>>(base_);
    }

  private:
    template<class... UTypes>
    friend class tuple;

    template<size_t i>
    __TUPLE_ANNOTATION
    const typename std::tuple_element<i,tuple>::type& const_get() const
    {
      return base_.template const_get<i>();
    }

    template<size_t i>
    __TUPLE_ANNOTATION
    typename std::tuple_element<i,tuple>::type& mutable_get()
    {
      return base_.template mutable_get<i>();
    }

    using base_type = __tuple_base<__tuple_make_index_sequence<sizeof...(Types)>, Types...>;
    base_type base_; 

  public:
    template<size_t i, class... UTypes>
    friend __TUPLE_ANNOTATION
    typename std::tuple_element<i, __TUPLE_NAMESPACE::tuple<UTypes...>>::type &
    std::get(__TUPLE_NAMESPACE::tuple<UTypes...>& t);


    template<size_t i, class... UTypes>
    friend __TUPLE_ANNOTATION
    const typename std::tuple_element<i, __TUPLE_NAMESPACE::tuple<UTypes...>>::type &
    std::get(const __TUPLE_NAMESPACE::tuple<UTypes...>& t);


    template<size_t i, class... UTypes>
    friend __TUPLE_ANNOTATION
    typename std::tuple_element<i, __TUPLE_NAMESPACE::tuple<UTypes...>>::type &&
    std::get(__TUPLE_NAMESPACE::tuple<UTypes...>&& t);
};


template<>
class tuple<>
{
  public:
    __TUPLE_ANNOTATION
    void swap(tuple&){}
};


template<class... Types>
__TUPLE_ANNOTATION
void swap(tuple<Types...>& a, tuple<Types...>& b)
{
  a.swap(b);
}


template<class... Types>
__TUPLE_ANNOTATION
tuple<typename std::decay<Types>::type...> make_tuple(Types&&... args)
{
  return tuple<typename std::decay<Types>::type...>(std::forward<Types>(args)...);
}


template<class... Types>
__TUPLE_ANNOTATION
tuple<Types&...> tie(Types&... args)
{
  return tuple<Types&...>(args...);
}


template<class... Args>
__TUPLE_ANNOTATION
__TUPLE_NAMESPACE::tuple<Args&&...> forward_as_tuple(Args&&... args)
{
  return __TUPLE_NAMESPACE::tuple<Args&&...>(std::forward<Args>(args)...);
}


} // end namespace


// implement std::get()
namespace std
{


template<size_t i, class... UTypes>
__TUPLE_ANNOTATION
typename std::tuple_element<i, __TUPLE_NAMESPACE::tuple<UTypes...>>::type &
  get(__TUPLE_NAMESPACE::tuple<UTypes...>& t)
{
  return t.template mutable_get<i>();
}


template<size_t i, class... UTypes>
__TUPLE_ANNOTATION
const typename std::tuple_element<i, __TUPLE_NAMESPACE::tuple<UTypes...>>::type &
  get(const __TUPLE_NAMESPACE::tuple<UTypes...>& t)
{
  return t.template const_get<i>();
}


template<size_t i, class... UTypes>
__TUPLE_ANNOTATION
typename std::tuple_element<i, __TUPLE_NAMESPACE::tuple<UTypes...>>::type &&
  get(__TUPLE_NAMESPACE::tuple<UTypes...>&& t)
{
  using type = typename std::tuple_element<i, __TUPLE_NAMESPACE::tuple<UTypes...>>::type;

  auto&& leaf = static_cast<__TUPLE_NAMESPACE::__tuple_leaf<i,type>&&>(t.base_);

  return static_cast<type&&>(leaf.mutable_get());
}


} // end std


#ifdef __TUPLE_ANNOTATION_NEEDS_UNDEF
#undef __TUPLE_ANNOTATION
#undef __TUPLE_ANNOTATION_NEEDS_UNDEF
#endif

#ifdef __TUPLE_NAMESPACE_NEEDS_UNDEF
#undef __TUPLE_NAMESPACE
#undef __TUPLE_NAMESPACE_NEEDS_UNDEF
#endif

