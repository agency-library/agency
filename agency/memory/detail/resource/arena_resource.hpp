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
#include <agency/memory/detail/resource/null_resource.hpp>
#include <new>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <type_traits>

namespace agency
{
namespace detail
{


// arena_resource is a C++ "memory resource" which
// allocates memory from a compile time-sized buffer
// it is based on Howard Hinnant's arena template here:
// https://howardhinnant.github.io/short_alloc.h
template<std::size_t N, std::size_t alignment = alignof(std::max_align_t)>
class arena_resource
{
  alignas(alignment) char buf_[N];
  unsigned int index_of_first_free_byte_;

  public:
    __AGENCY_ANNOTATION
    arena_resource() noexcept : index_of_first_free_byte_(0) {}
    
    __AGENCY_ANNOTATION
    arena_resource(const arena_resource&) = delete;

    __AGENCY_ANNOTATION
    arena_resource& operator=(const arena_resource&) = delete;

    __AGENCY_ANNOTATION
    void* allocate(std::size_t n) noexcept
    {
      std::size_t aligned_n = align_up(n);
      if(aligned_n > num_remaining_bytes())
      {
        return nullptr;
      }

      char* r = buf_ + index_of_first_free_byte_;
      index_of_first_free_byte_ += aligned_n;
      return r;
    }

    __AGENCY_ANNOTATION
    void deallocate(void* p_, std::size_t n) noexcept
    {
      unsigned int index_of_p = reinterpret_cast<char*>(p_) - buf_;
      std::size_t aligned_n = align_up(n);

      if(index_of_p == index_of_first_free_byte_ - aligned_n)
      {
        index_of_first_free_byte_ = index_of_p;
      }
    }

    __AGENCY_ANNOTATION
    static constexpr std::size_t size() noexcept
    {
      return N;
    }

    __AGENCY_ANNOTATION
    void reset() noexcept
    {
      index_of_first_free_byte_ = 0;
    }

    __AGENCY_ANNOTATION
    bool owns(void* ptr, std::size_t) const noexcept
    {
      int index_of_ptr = reinterpret_cast<char*>(ptr) - buf_;

      return index_of_ptr >= 0 && index_of_first_free_byte_ >= static_cast<unsigned int>(index_of_ptr);
    }

  private:
    __AGENCY_ANNOTATION
    static std::size_t align_up(std::size_t n) noexcept
    {
      return (n + (alignment-1)) & ~(alignment-1);
    }

    __AGENCY_ANNOTATION
    const char* end_of_buffer() const noexcept
    {
      return buf_ + size();
    }

    __AGENCY_ANNOTATION
    std::size_t num_remaining_bytes() const noexcept
    {
      return N - index_of_first_free_byte_;
    }
};


// in the special case of zero size, use null_resource for efficiency
// it avoids calling align_up() in allocate()
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
    void reset() noexcept
    {
    }

    __AGENCY_ANNOTATION
    constexpr bool owns(void*, std::size_t) const noexcept
    {
      return false;
    }
};


} // end detail
} // end agency

