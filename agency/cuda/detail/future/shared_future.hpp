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


// XXX TODO port this class's members to __host__ __device__
//          by introducing a CUDA-compatible implementation of shared_ptr
template<class T>
class shared_future
{
  private:
    std::shared_ptr<agency::cuda::future<T>> underlying_future_;

  public:
    shared_future() = default;

    shared_future(const shared_future&) = default;

    shared_future(agency::cuda::future<T>&& other)
      : underlying_future_(std::make_shared<agency::cuda::future<T>>(std::move(other)))
    {}

    shared_future(shared_future&& other) = default;

    ~shared_future() = default;

    shared_future& operator=(const shared_future& other) = default;

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
               agency::cuda::detail::is_constructible_or_void<T,Args...>::value
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

    template<class Function, class Factory, class Shape, class IndexFunction, class OuterFactory, class InnerFactory>
    agency::cuda::future<agency::detail::result_of_t<Factory(Shape)>>
      old_bulk_then(Function f, Factory result_factory, Shape shape, IndexFunction index_function, OuterFactory outer_factory, InnerFactory inner_factory, agency::cuda::device_id device)
    {
      return underlying_future_->old_bulk_then_and_leave_valid(f, result_factory, shape, index_function, outer_factory, inner_factory, device);
    }

    template<class Function, class Shape, class IndexFunction, class ResultFactory, class OuterFactory, class InnerFactory>
    agency::cuda::future<agency::detail::result_of_t<ResultFactory()>>
      bulk_then(Function f, Shape shape, IndexFunction index_function, ResultFactory result_factory, OuterFactory outer_factory, InnerFactory inner_factory, agency::cuda::device_id device)
    {
      return underlying_future_->bulk_then_and_leave_valid(f, shape, index_function, result_factory, outer_factory, inner_factory, device);
    }
};


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

