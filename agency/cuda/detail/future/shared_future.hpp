/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/future/future.hpp>
#include <agency/detail/type_traits.hpp>
#include <type_traits>
#include <memory>

namespace agency
{
namespace cuda
{


template<class T>
class shared_future;

namespace detail
{


template<class T>
agency::cuda::future<T>& underlying_future(shared_future<T>& future);

 
} // end detail


// XXX TODO port this class's members to __host__ __device__
//          by introducing a CUDA-compatible implementation of shared_ptr
template<class T>
class shared_future
{
  private:
    std::shared_ptr<agency::cuda::future<T>> underlying_future_;

    agency::cuda::future<T>& underlying_future()
    {
      return *underlying_future_;
    }

    // friend underlying_future to give it access to underlying_future()
    template<class U>
    friend agency::cuda::future<U>& detail::underlying_future(shared_future<U>& future);

  public:
    __agency_exec_check_disable__
    shared_future() = default;

    __agency_exec_check_disable__
    shared_future(const shared_future&) = default;

    shared_future(agency::cuda::future<T>&& other)
      : underlying_future_(std::make_shared<agency::cuda::future<T>>(std::move(other)))
    {}

    __agency_exec_check_disable__
    shared_future(shared_future&& other) = default;

    __agency_exec_check_disable__
    ~shared_future() = default;

    __agency_exec_check_disable__
    shared_future& operator=(const shared_future& other) = default;

    __agency_exec_check_disable__
    shared_future& operator=(shared_future&& other) = default;

    bool valid() const
    {
      return underlying_future_ && underlying_future_->valid();
    }

    bool is_ready() const
    {
      return underlying_future_ && underlying_future_->is_ready();
    }

    void wait() const
    {
      underlying_future_->wait();
    }

    auto get() ->
      decltype(underlying_future_->get_ref())
    {
      return underlying_future_->get_ref();
    } // end get()

    template<class... Args,
             class = typename std::enable_if<
               agency::detail::is_constructible_or_void<T,Args...>::value
             >::type>
    static shared_future make_ready(Args&&... args)
    {
      return agency::cuda::future<T>::make_ready(std::forward<Args>(args)...).share();
    }

    template<class Function>
    agency::cuda::future<
      agency::detail::result_of_continuation_t<
        typename std::decay<Function>::type,
        shared_future
      >
    >
      then(Function f)
    {
      // XXX what if there are no shared_futures by the time the continuation runs?
      //     who owns the data?
      //     it seems like we need to introduce a copy of this shared_future into
      //     a continuation dependent on the next_event

      return underlying_future_->then_and_leave_valid(f);
    }
};


namespace detail
{


template<class T>
agency::cuda::future<T>& underlying_future(shared_future<T>& future)
{
  return future.underlying_future();
}

 
} // end detail


// implement async_future<T>::share() here because this implementation
// requires the definition of one of shared_future<T>'s ctors
template<class T>
shared_future<T> async_future<T>::share()
{
  return shared_future<T>(std::move(*this));
}


// implement future<T>::share() here because this implementation
// requires the definition of one of shared_future<T>'s ctors
template<class T>
shared_future<T> future<T>::share()
{
  return shared_future<T>(std::move(*this));
}


} // end cuda
} // end agency

