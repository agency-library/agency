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

#include <new>
#include <type_traits>
#include <utility>


namespace agency
{
namespace cuda
{
namespace detail
{


template<typename T>
  class uninitialized
{
  private:
    typename std::aligned_storage<
      sizeof(T),
      std::alignment_of<T>::value
    >::type storage[1];

    __host__ __device__
    const T* ptr() const
    {
      const void *result = storage;
      return reinterpret_cast<const T*>(result);
    }

    __host__ __device__
    T* ptr()
    {
      void *result = storage;
      return reinterpret_cast<T*>(result);
    }

  public:
    // copy assignment
    __host__ __device__
    uninitialized<T> &operator=(const T &other)
    {
      T& self = *this;
      self = other;
      return *this;
    }

    __host__ __device__
    T& get()
    {
      return *ptr();
    }

    __host__ __device__
    const T& get() const
    {
      return *ptr();
    }

    __host__ __device__
    operator T& ()
    {
      return get();
    }

    __host__ __device__
    operator const T&() const
    {
      return get();
    }

    template<class... Args>
    __host__ __device__
    void construct(Args&&... args)
    {
      ::new(ptr()) T(std::forward<Args>(args)...);
    }

    __host__ __device__
    void destroy()
    {
      T& self = *this;
      self.~T();
    }
};


} // end detail
} // end cuda
} // end agency

