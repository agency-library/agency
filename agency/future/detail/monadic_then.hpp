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
#include <agency/detail/type_traits.hpp>
#include <agency/future/future_traits/detail/has_then_member.hpp>
#include <utility>
#include <future>


namespace agency
{
namespace detail
{


template<class T, class Function>
std::future<detail::result_of_t<Function(T&)>>
  monadic_then(std::future<T>& fut, std::launch policy, Function&& f)
{
  return std::async(policy, [](std::future<T>&& fut, Function&& f)
  {
    T arg = fut.get();
    return std::forward<Function>(f)(arg);
  },
  std::move(fut),
  std::forward<Function>(f)
  );
}


template<class Function>
std::future<detail::result_of_t<Function()>>
  monadic_then(std::future<void>& fut, std::launch policy, Function&& f)
{
  return std::async(policy, [](std::future<void>&& fut, Function&& f)
  {
    fut.get();
    return std::forward<Function>(f)();
  },
  std::move(fut),
  std::forward<Function>(f)
  );
}


template<class T, class Function>
auto monadic_then(std::future<T>& fut, Function&& f) ->
  decltype(detail::monadic_then(fut, std::launch::async | std::launch::deferred, std::forward<Function>(f)))
{
  return detail::monadic_then(fut, std::launch::async | std::launch::deferred, std::forward<Function>(f));
}


// monadic_then() for std::shared_future
template<class T, class Function>
std::future<detail::result_of_t<Function(T&)>>
  monadic_then(std::shared_future<T>& fut, std::launch policy, Function&& f)
{
  return std::async(policy, [](std::shared_future<T>&& fut, Function&& f)
  {
    T& arg = const_cast<T&>(fut.get());
    return std::forward<Function>(f)(arg);
  },
  std::move(fut),
  std::forward<Function>(f)
  );
}


template<class Function>
std::future<detail::result_of_t<Function()>>
  monadic_then(std::shared_future<void>& fut, std::launch policy, Function&& f)
{
  return std::async(policy, [](std::shared_future<void>&& fut, Function&& f)
  {
    fut.get();
    return std::forward<Function>(f)();
  },
  std::move(fut),
  std::forward<Function>(f)
  );
}


template<class T, class Function>
auto monadic_then(std::shared_future<T>& fut, Function&& f) ->
  decltype(detail::monadic_then(fut, std::launch::async | std::launch::deferred, std::forward<Function>(f)))
{
  return detail::monadic_then(fut, std::launch::async | std::launch::deferred, std::forward<Function>(f));
}


// no adaptation is required if the given future's .then() is already the monadic form
template<class Future, class Function,
         __AGENCY_REQUIRES(is_future<Future>::value),
         __AGENCY_REQUIRES(has_then_member<Future, Function&&>::value)
        >
auto monadic_then(Future& fut, Function&& f) ->
  decltype(fut.then(std::forward<Function>(f)))
{
  return fut.then(std::forward<Function>(f));
}


} // end detail
} // end agency

