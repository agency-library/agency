#pragma once

// The MIT License (MIT)
// 
// Copyright (c) 2015 Howard Hinnant
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <agency/detail/config.hpp>
#include <agency/detail/memory/null_resource.hpp>
#include <new>
#include <cassert>
#include <cstddef>
#include <cstdio>

namespace agency
{
namespace detail
{
namespace arena_resource_detail
{


__AGENCY_ANNOTATION
inline void throw_bad_alloc()
{
#ifdef __CUDA_ARCH__
  printf("bad_alloc\n");
  assert(0);
#else
  throw std::bad_alloc();
#endif
}


} // end arena_resource_detail


// arena_resource is a C++ "memory resource" which
// allocates memory from a compile time-sized buffer
// it is based on Howard Hinnant's arena template here:
// https://howardhinnant.github.io/short_alloc.h
template<std::size_t N, std::size_t alignment = alignof(std::max_align_t)>
class arena_resource
{
  alignas(alignment) char buf_[N];
  char* ptr_;

  public:
    __AGENCY_ANNOTATION
    ~arena_resource() {ptr_ = nullptr;}

    __AGENCY_ANNOTATION
    arena_resource() noexcept : ptr_(buf_) {}

    __AGENCY_ANNOTATION
    arena_resource(const arena_resource&) = delete;

    __AGENCY_ANNOTATION
    arena_resource& operator=(const arena_resource&) = delete;

    __AGENCY_ANNOTATION
    void* allocate(std::size_t n)
    {
      auto const aligned_n = align_up(n);
      if(aligned_n > static_cast<decltype(aligned_n)>(buf_ + N - ptr_))
      {
        // XXX should we throw? we might want to fail and return 0
        //     maybe we could introduce a separate allocate_or_fail() function
        arena_resource_detail::throw_bad_alloc();
      }

      char* r = ptr_;
      ptr_ += aligned_n;
      return r;
    }

    __AGENCY_ANNOTATION
    void deallocate(void* p_, std::size_t n) noexcept
    {
      char* p = reinterpret_cast<char*>(p_);

      n = align_up(n);
      if(p + n == ptr_)
      {
        ptr_ = p;
      }
    }

    __AGENCY_ANNOTATION
    static constexpr std::size_t size() noexcept
    {
      return N;
    }

    __AGENCY_ANNOTATION
    std::size_t used() const noexcept
    {
      return static_cast<std::size_t>(ptr_ - buf_);
    }

    __AGENCY_ANNOTATION
    void reset() noexcept
    {
      ptr_ = buf_;
    }

  private:
    __AGENCY_ANNOTATION
    static std::size_t align_up(std::size_t n) noexcept
    {
      return (n + (alignment-1)) & ~(alignment-1);
    }
};


// in the special case of zero size, use null_resource for efficiency
// it avoids calling align_up() in allocate() and does not throw an exception
// XXX we should avoid throwing in arena_resource::allocate() above for consistency
template<std::size_t alignment>
class arena_resource<0,alignment> : public null_resource
{
  public:
    using null_resource::null_resource;

    __AGENCY_ANNOTATION
    static constexpr std::size_t size() noexcept
    {
      return 0;
    }

    __AGENCY_ANNOTATION
    static constexpr std::size_t used() noexcept
    {
      return 0;
    }

    __AGENCY_ANNOTATION
    void reset() noexcept
    {
    }
};


} // end detail
} // end agency

