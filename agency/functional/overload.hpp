// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include <type_traits>
#include <utility>


namespace agency
{
namespace detail
{


template<class Function1>
struct overloaded1 : Function1
{
  overloaded1() = default;
  overloaded1(const overloaded1&) = default;

  template<class OtherFunction1>
  __AGENCY_ANNOTATION
  constexpr overloaded1(OtherFunction1&& f1)
    : Function1(std::forward<OtherFunction1>(f1))
  {}

  using Function1::operator();
};


template<class Function1, class Function2>
struct overloaded2 : Function1, Function2
{
  overloaded2() = default;
  overloaded2(const overloaded2&) = default;

  template<class OtherFunction1, class OtherFunction2>
  __AGENCY_ANNOTATION
  constexpr overloaded2(OtherFunction1&& f1, OtherFunction2&& f2)
    : Function1(std::forward<OtherFunction1>(f1)),
      Function2(std::forward<OtherFunction2>(f2))
  {}

  using Function1::operator();
  using Function2::operator();
};


template<class Function1, class Function2, class Function3>
struct overloaded3 : Function1, Function2, Function3
{
  overloaded3() = default;
  overloaded3(const overloaded3&) = default;

  template<class OtherFunction1, class OtherFunction2, class OtherFunction3>
  __AGENCY_ANNOTATION
  constexpr overloaded3(OtherFunction1&& f1, OtherFunction2&& f2, OtherFunction3&& f3)
    : Function1(std::forward<OtherFunction1>(f1)),
      Function2(std::forward<OtherFunction2>(f2)),
      Function3(std::forward<OtherFunction3>(f3))
  {}

  using Function1::operator();
  using Function2::operator();
  using Function3::operator();
};


template<class Function1, class Function2, class Function3, class Function4>
struct overloaded4 : Function1, Function2, Function3, Function4
{
  overloaded4() = default;
  overloaded4(const overloaded4&) = default;

  template<class OtherFunction1, class OtherFunction2, class OtherFunction3, class OtherFunction4>
  __AGENCY_ANNOTATION
  constexpr overloaded4(OtherFunction1&& f1, OtherFunction2&& f2, OtherFunction3& f3, OtherFunction4& f4)
    : Function1(std::forward<OtherFunction1>(f1)),
      Function2(std::forward<OtherFunction2>(f2)),
      Function3(std::forward<OtherFunction3>(f3)),
      Function4(std::forward<OtherFunction4>(f4))
  {}

  using Function1::operator();
  using Function2::operator();
  using Function3::operator();
  using Function4::operator();
};


template<class Function1, class Function2, class Function3, class Function4, class Function5>
struct overloaded5 : Function1, Function2, Function3, Function4, Function5
{
  overloaded5() = default;
  overloaded5(const overloaded5&) = default;

  template<class OtherFunction1, class OtherFunction2, class OtherFunction3, class OtherFunction4, class OtherFunction5>
  __AGENCY_ANNOTATION
  constexpr overloaded5(OtherFunction1&& f1, OtherFunction2&& f2, OtherFunction3&& f3, OtherFunction4&& f4, OtherFunction5&& f5)
    : Function1(std::forward<OtherFunction1>(f1)),
      Function2(std::forward<OtherFunction2>(f2)),
      Function3(std::forward<OtherFunction3>(f3)),
      Function4(std::forward<OtherFunction4>(f4)),
      Function5(std::forward<OtherFunction5>(f5))
  {}

  using Function1::operator();
  using Function2::operator();
  using Function3::operator();
  using Function4::operator();
  using Function5::operator();
};


template<class Function1, class Function2, class Function3, class Function4, class Function5, class Function6>
struct overloaded6 : Function1, Function2, Function3, Function4, Function5, Function6
{
  overloaded6() = default;
  overloaded6(const overloaded6&) = default;

  template<class OtherFunction1, class OtherFunction2, class OtherFunction3, class OtherFunction4, class OtherFunction5, class OtherFunction6>
  __AGENCY_ANNOTATION
  constexpr overloaded6(OtherFunction1&& f1, OtherFunction2&& f2, OtherFunction3&& f3, OtherFunction4&& f4, OtherFunction5&& f5, OtherFunction6&& f6)
    : Function1(std::forward<OtherFunction1>(f1)),
      Function2(std::forward<OtherFunction2>(f2)),
      Function3(std::forward<OtherFunction3>(f3)),
      Function4(std::forward<OtherFunction4>(f4)),
      Function5(std::forward<OtherFunction5>(f5)),
      Function6(std::forward<OtherFunction6>(f6))
  {}

  using Function1::operator();
  using Function2::operator();
  using Function3::operator();
  using Function4::operator();
  using Function5::operator();
  using Function6::operator();
};


template<class Function1, class Function2, class Function3, class Function4, class Function5, class Function6, class Function7>
struct overloaded7 : Function1, Function2, Function3, Function4, Function5, Function6, Function7
{
  overloaded7() = default;
  overloaded7(const overloaded7&) = default;

  template<class OtherFunction1, class OtherFunction2, class OtherFunction3, class OtherFunction4, class OtherFunction5, class OtherFunction6, class OtherFunction7>
  __AGENCY_ANNOTATION
  constexpr overloaded7(OtherFunction1&& f1, OtherFunction2&& f2, OtherFunction3&& f3, OtherFunction4&& f4, OtherFunction5&& f5, OtherFunction6&& f6, OtherFunction7&& f7)
    : Function1(std::forward<OtherFunction1>(f1)),
      Function2(std::forward<OtherFunction2>(f2)),
      Function3(std::forward<OtherFunction3>(f3)),
      Function4(std::forward<OtherFunction4>(f4)),
      Function5(std::forward<OtherFunction5>(f5)),
      Function6(std::forward<OtherFunction6>(f6)),
      Function7(std::forward<OtherFunction7>(f7))
  {}

  using Function1::operator();
  using Function2::operator();
  using Function3::operator();
  using Function4::operator();
  using Function5::operator();
  using Function6::operator();
  using Function7::operator();
};


template<class Function1, class Function2, class Function3, class Function4, class Function5, class Function6, class Function7, class Function8>
struct overloaded8 : Function1, Function2, Function3, Function4, Function5, Function6, Function7, Function8
{
  overloaded8() = default;
  overloaded8(const overloaded8&) = default;

  template<class OtherFunction1, class OtherFunction2, class OtherFunction3, class OtherFunction4, class OtherFunction5, class OtherFunction6, class OtherFunction7, class OtherFunction8>
  __AGENCY_ANNOTATION
  constexpr overloaded8(OtherFunction1&& f1, OtherFunction2&& f2, OtherFunction3&& f3, OtherFunction4&& f4, OtherFunction5&& f5, OtherFunction6&& f6, OtherFunction7&& f7, OtherFunction8&& f8)
    : Function1(std::forward<OtherFunction1>(f1)),
      Function2(std::forward<OtherFunction2>(f2)),
      Function3(std::forward<OtherFunction3>(f3)),
      Function4(std::forward<OtherFunction4>(f4)),
      Function5(std::forward<OtherFunction5>(f5)),
      Function6(std::forward<OtherFunction6>(f6)),
      Function7(std::forward<OtherFunction7>(f7)),
      Function8(std::forward<OtherFunction8>(f8))
  {}

  using Function1::operator();
  using Function2::operator();
  using Function3::operator();
  using Function4::operator();
  using Function5::operator();
  using Function6::operator();
  using Function7::operator();
  using Function8::operator();
};


template<class Function1, class Function2, class Function3, class Function4, class Function5, class Function6, class Function7, class Function8, class Function9>
struct overloaded9 : Function1, Function2, Function3, Function4, Function5, Function6, Function7, Function8, Function9
{
  overloaded9() = default;
  overloaded9(const overloaded9&) = default;

  template<class OtherFunction1, class OtherFunction2, class OtherFunction3, class OtherFunction4, class OtherFunction5, class OtherFunction6, class OtherFunction7, class OtherFunction8, class OtherFunction9>
  __AGENCY_ANNOTATION
  constexpr overloaded9(OtherFunction1&& f1, OtherFunction2&& f2, OtherFunction3&& f3, OtherFunction4&& f4, OtherFunction5&& f5, OtherFunction6&& f6, OtherFunction7&& f7, OtherFunction8&& f8, OtherFunction9&& f9)
    : Function1(std::forward<OtherFunction1>(f1)),
      Function2(std::forward<OtherFunction2>(f2)),
      Function3(std::forward<OtherFunction3>(f3)),
      Function4(std::forward<OtherFunction4>(f4)),
      Function5(std::forward<OtherFunction5>(f5)),
      Function6(std::forward<OtherFunction6>(f6)),
      Function7(std::forward<OtherFunction7>(f7)),
      Function8(std::forward<OtherFunction8>(f8)),
      Function9(std::forward<OtherFunction9>(f9))
  {}

  using Function1::operator();
  using Function2::operator();
  using Function3::operator();
  using Function4::operator();
  using Function5::operator();
  using Function6::operator();
  using Function7::operator();
  using Function8::operator();
  using Function9::operator();
};


template<class Function1, class Function2, class Function3, class Function4, class Function5, class Function6, class Function7, class Function8, class Function9, class Function10>
struct overloaded10 : Function1, Function2, Function3, Function4, Function5, Function6, Function7, Function8, Function9, Function10
{
  overloaded10() = default;
  overloaded10(const overloaded10&) = default;

  template<class OtherFunction1, class OtherFunction2, class OtherFunction3, class OtherFunction4, class OtherFunction5, class OtherFunction6, class OtherFunction7, class OtherFunction8, class OtherFunction9, class OtherFunction10>
  __AGENCY_ANNOTATION
  constexpr overloaded10(OtherFunction1&& f1, OtherFunction2&& f2, OtherFunction3&& f3, OtherFunction4&& f4, OtherFunction5&& f5, OtherFunction6&& f6, OtherFunction7&& f7, OtherFunction8&& f8, OtherFunction9&& f9, OtherFunction10&& f10)
    : Function1(std::forward<OtherFunction1>(f1)),
      Function2(std::forward<OtherFunction2>(f2)),
      Function3(std::forward<OtherFunction3>(f3)),
      Function4(std::forward<OtherFunction4>(f4)),
      Function5(std::forward<OtherFunction5>(f5)),
      Function6(std::forward<OtherFunction6>(f6)),
      Function7(std::forward<OtherFunction7>(f7)),
      Function8(std::forward<OtherFunction8>(f8)),
      Function9(std::forward<OtherFunction8>(f9)),
      Function10(std::forward<OtherFunction10>(f10))
  {}

  using Function1::operator();
  using Function2::operator();
  using Function3::operator();
  using Function4::operator();
  using Function5::operator();
  using Function6::operator();
  using Function7::operator();
  using Function8::operator();
  using Function9::operator();
  using Function10::operator();
};


} // end detail


template<class Function1>
__AGENCY_ANNOTATION
constexpr detail::overloaded1<typename std::decay<Function1>::type>
  overload(Function1&& f1)
{
  return {std::forward<Function1>(f1)};
}


template<class Function1, class Function2>
__AGENCY_ANNOTATION
constexpr detail::overloaded2<
  typename std::decay<Function1>::type,
  typename std::decay<Function2>::type
>
  overload(Function1&& f1, Function2&& f2)
{
  return {std::forward<Function1>(f1), std::forward<Function2>(f2)};
}


template<class Function1, class Function2, class Function3>
__AGENCY_ANNOTATION
constexpr detail::overloaded3<
  typename std::decay<Function1>::type,
  typename std::decay<Function2>::type,
  typename std::decay<Function3>::type
>
  overload(Function1&& f1, Function2&& f2, Function3&& f3)
{
  return {std::forward<Function1>(f1), std::forward<Function2>(f2), std::forward<Function3>(f3)};
}


template<class Function1, class Function2, class Function3, class Function4>
__AGENCY_ANNOTATION
constexpr detail::overloaded4<
  typename std::decay<Function1>::type,
  typename std::decay<Function2>::type,
  typename std::decay<Function3>::type,
  typename std::decay<Function4>::type
>
  overload(Function1&& f1, Function2&& f2, Function3&& f3, Function4&& f4)
{
  return {std::forward<Function1>(f1), std::forward<Function2>(f2), std::forward<Function3>(f3), std::forward<Function4>(f4)};
}


template<class Function1, class Function2, class Function3, class Function4, class Function5>
__AGENCY_ANNOTATION
constexpr detail::overloaded5<
  typename std::decay<Function1>::type,
  typename std::decay<Function2>::type,
  typename std::decay<Function3>::type,
  typename std::decay<Function4>::type,
  typename std::decay<Function5>::type
>
  overload(Function1&& f1, Function2&& f2, Function3&& f3, Function4&& f4, Function5&& f5)
{
  return {std::forward<Function1>(f1), std::forward<Function2>(f2), std::forward<Function3>(f3), std::forward<Function4>(f4), std::forward<Function5>(f5)};
}


template<class Function1, class Function2, class Function3, class Function4, class Function5, class Function6>
__AGENCY_ANNOTATION
constexpr detail::overloaded6<
  typename std::decay<Function1>::type,
  typename std::decay<Function2>::type,
  typename std::decay<Function3>::type,
  typename std::decay<Function4>::type,
  typename std::decay<Function5>::type,
  typename std::decay<Function6>::type
>
  overload(Function1&& f1, Function2&& f2, Function3&& f3, Function4&& f4, Function5&& f5, Function6&& f6)
{
  return {std::forward<Function1>(f1), std::forward<Function2>(f2), std::forward<Function3>(f3), std::forward<Function4>(f4), std::forward<Function5>(f5), std::forward<Function6>(f6)};
}


template<class Function1, class Function2, class Function3, class Function4, class Function5, class Function6, class Function7>
__AGENCY_ANNOTATION
constexpr detail::overloaded7<
  typename std::decay<Function1>::type,
  typename std::decay<Function2>::type,
  typename std::decay<Function3>::type,
  typename std::decay<Function4>::type,
  typename std::decay<Function5>::type,
  typename std::decay<Function6>::type,
  typename std::decay<Function7>::type
>
  overload(Function1&& f1, Function2&& f2, Function3&& f3, Function4&& f4, Function5&& f5, Function6&& f6, Function7&& f7)
{
  return {std::forward<Function1>(f1), std::forward<Function2>(f2), std::forward<Function3>(f3), std::forward<Function4>(f4), std::forward<Function5>(f5), std::forward<Function6>(f6), std::forward<Function7>(f7)};
}


template<class Function1, class Function2, class Function3, class Function4, class Function5, class Function6, class Function7, class Function8>
__AGENCY_ANNOTATION
constexpr detail::overloaded8<
  typename std::decay<Function1>::type,
  typename std::decay<Function2>::type,
  typename std::decay<Function3>::type,
  typename std::decay<Function4>::type,
  typename std::decay<Function5>::type,
  typename std::decay<Function6>::type,
  typename std::decay<Function7>::type,
  typename std::decay<Function8>::type
>
  overload(Function1&& f1, Function2&& f2, Function3&& f3, Function4&& f4, Function5&& f5, Function6&& f6, Function7&& f7, Function8&& f8)
{
  return {std::forward<Function1>(f1), std::forward<Function2>(f2), std::forward<Function3>(f3), std::forward<Function4>(f4), std::forward<Function5>(f5), std::forward<Function6>(f6), std::forward<Function7>(f7), std::forward<Function8>(f8)};
}


template<class Function1, class Function2, class Function3, class Function4, class Function5, class Function6, class Function7, class Function8, class Function9>
__AGENCY_ANNOTATION
constexpr detail::overloaded9<
  typename std::decay<Function1>::type,
  typename std::decay<Function2>::type,
  typename std::decay<Function3>::type,
  typename std::decay<Function4>::type,
  typename std::decay<Function5>::type,
  typename std::decay<Function6>::type,
  typename std::decay<Function7>::type,
  typename std::decay<Function8>::type,
  typename std::decay<Function9>::type
>
  overload(Function1&& f1, Function2&& f2, Function3&& f3, Function4&& f4, Function5&& f5, Function6&& f6, Function7&& f7, Function8&& f8, Function9&& f9)
{
  return {std::forward<Function1>(f1), std::forward<Function2>(f2), std::forward<Function3>(f3), std::forward<Function4>(f4), std::forward<Function5>(f5), std::forward<Function6>(f6), std::forward<Function7>(f7), std::forward<Function8>(f8), std::forward<Function9>(f9)};
}


template<class Function1, class Function2, class Function3, class Function4, class Function5, class Function6, class Function7, class Function8, class Function9, class Function10>
__AGENCY_ANNOTATION
constexpr detail::overloaded10<
  typename std::decay<Function1>::type,
  typename std::decay<Function2>::type,
  typename std::decay<Function3>::type,
  typename std::decay<Function4>::type,
  typename std::decay<Function5>::type,
  typename std::decay<Function6>::type,
  typename std::decay<Function7>::type,
  typename std::decay<Function8>::type,
  typename std::decay<Function9>::type,
  typename std::decay<Function10>::type
>
  overload(Function1&& f1, Function2&& f2, Function3&& f3, Function4&& f4, Function5&& f5, Function6&& f6, Function7&& f7, Function8&& f8, Function9&& f9, Function10&& f10)
{
  return {std::forward<Function1>(f1), std::forward<Function2>(f2), std::forward<Function3>(f3), std::forward<Function4>(f4), std::forward<Function5>(f5), std::forward<Function6>(f6), std::forward<Function7>(f7), std::forward<Function8>(f8), std::forward<Function9>(f9), std::forward<Function10>(f10)};
}


} // end namespace agency

