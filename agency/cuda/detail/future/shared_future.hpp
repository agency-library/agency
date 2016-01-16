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
#include <type_traits>
#include <memory>

namespace agency
{
namespace cuda
{


template<class T>
class shared_future
{
  private:
    // XXX should we share a future<T> or a future_base<T> ?
    std::shared_ptr<future<T>> underlying_future_;

    detail::event& event()
    {
      return underlying_future_->event();
    } // end event()

    auto data() const ->
      decltype(underlying_future_->data())
    {
      return underlying_future_->data();
    }

  public:
    shared_future() = default;

    shared_future(const shared_future&) = default;

    shared_future(future<T>&& other)
      : underlying_future_(std::make_shared<future<T>>(std::move(other)))
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

  private:
    void convert_data_to_result_of_get(agency::detail::unit_ptr)
    {
    }

    template<class U>
    const U& convert_data_to_result_of_get(U* ptr)
    {
      return *ptr;
    }

  public:
    auto get() ->
      decltype(this->convert_data_to_result_of_get(this->data()))
    {
      wait();
      return this->convert_data_to_result_of_get(this->data());
    } // end get()

    template<class... Args,
             class = typename std::enable_if<
               detail::is_constructible_or_void<T,Args...>::value
             >::type>
    static shared_future make_ready(Args&&... args)
    {
      return future<T>::make_ready(std::forward<Args>(args)...).share();
    }

    template<class Function>
    future<
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

      // create state for the continuation's result
      using result_type = agency::detail::result_of_continuation_t<typename std::decay<Function>::type,shared_future>;
      detail::asynchronous_state<result_type> result_state(agency::detail::construct_not_ready);

      // tuple up f's input state
      auto unfiltered_pointer_tuple = agency::detail::make_tuple(data());

      // filter void states
      auto pointer_tuple = agency::detail::tuple_filter<detail::element_type_is_not_unit>(unfiltered_pointer_tuple);
      
      // make a function implementing the continuation
      auto continuation = detail::make_continuation(std::forward<Function>(f), result_state.data(), pointer_tuple);

      // launch the continuation
      detail::event next_event = event().then(std::move(continuation), dim3{1}, dim3{1}, 0);

      // return the continuation's future
      return future<result_type>(std::move(next_event), std::move(result_state));
    }
};


// implement future<T>::share() here because this implementation
// requires the definition of one of shared_future<T>'s ctors
template<class T>
shared_future<T> future<T>::share()
{
  return shared_future<T>(std::move(*this));
}


} // end cuda
} // end agency

