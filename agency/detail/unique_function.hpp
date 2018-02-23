#pragma once

#include <agency/detail/config.hpp>
#include <agency/memory/detail/unique_ptr.hpp>
#include <stdexcept>
#include <cassert>
#include <utility>
#include <type_traits>
#include <memory>


namespace agency
{
namespace detail
{


class bad_function_call : public std::exception
{
  public:
    virtual const char* what() const noexcept
    {
      return "bad_function_call: unique_function has no target";
    }
};


namespace unique_function_detail
{


__AGENCY_ANNOTATION
inline void throw_bad_function_call()
{
#ifdef __CUDA_ARCH__
  printf("bad_function_call: unique_function has no target\n");
  assert(0);
#else
  throw bad_function_call();
#endif
}


} // end unique_function_detail


template<class>
class unique_function;

template<class Result, class... Args>
class unique_function<Result(Args...)>
{
  public:
    using result_type = Result;

    unique_function() = default;

    __AGENCY_ANNOTATION
    unique_function(std::nullptr_t)
      : f_ptr_(nullptr)
    {}

    unique_function(unique_function&& other) = default;

    template<class Function>
    __AGENCY_ANNOTATION
    unique_function(Function&& f)
      : unique_function(std::allocator_arg, default_allocator<typename std::decay<Function>::type>{}, std::forward<Function>(f))
    {}

    template<class Alloc>
    __AGENCY_ANNOTATION
    unique_function(std::allocator_arg_t, const Alloc&, std::nullptr_t)
      : unique_function(nullptr)
    {}

    template<class Alloc>
    __AGENCY_ANNOTATION
    unique_function(std::allocator_arg_t, const Alloc& alloc)
      : unique_function(std::allocator_arg, alloc, nullptr)
    {}

    template<class Alloc>
    __AGENCY_ANNOTATION
    unique_function(std::allocator_arg_t, const Alloc&, unique_function&& other)
      : f_ptr_(std::move(other.f_ptr_))
    {}

    template<class Alloc, class Function>
    __AGENCY_ANNOTATION
    unique_function(std::allocator_arg_t, const Alloc& alloc, Function&& f)
      : f_ptr_(allocate_function_pointer(alloc, std::forward<Function>(f)))
    {}

    unique_function& operator=(unique_function&& other) = default;

    __AGENCY_ANNOTATION
    Result operator()(Args... args) const
    {
      if(!*this)
      {
        unique_function_detail::throw_bad_function_call();
      }

      return (*f_ptr_)(args...);
    }

    __AGENCY_ANNOTATION
    operator bool () const
    {
      return f_ptr_;
    }

  private:
    // this is the abstract base class for a type
    // which is both
    // 1. callable like a function and
    // 2. deallocates itself inside its destructor
    struct callable_self_deallocator_base
    {
      using self_deallocate_function_type = void(*)(callable_self_deallocator_base*);

      self_deallocate_function_type self_deallocate_function;

      template<class Function>
      __AGENCY_ANNOTATION
      callable_self_deallocator_base(Function callback)
        : self_deallocate_function(callback)
      {}

      __AGENCY_ANNOTATION
      virtual ~callable_self_deallocator_base()
      {
        self_deallocate_function(this);
      }

      __AGENCY_ANNOTATION
      virtual Result operator()(Args... args) const = 0;
    };

    template<class Function, class Alloc>
    struct callable : callable_self_deallocator_base
    {
      using super_t = callable_self_deallocator_base;
      using allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<callable>;

      mutable Function f_;

      __agency_exec_check_disable__
      ~callable() = default;

      __agency_exec_check_disable__
      template<class OtherFunction,
               class = typename std::enable_if<
                 std::is_constructible<Function,OtherFunction&&>::value
               >::type>
      __AGENCY_ANNOTATION
      callable(const Alloc&, OtherFunction&& f)
        : super_t(deallocate),
          f_(std::forward<OtherFunction>(f))
      {}

      __agency_exec_check_disable__
      __AGENCY_ANNOTATION
      virtual Result operator()(Args... args) const
      {
        return f_(args...);
      }

      __agency_exec_check_disable__
      __AGENCY_ANNOTATION
      static void deallocate(callable_self_deallocator_base* ptr)
      {
        // upcast to the right type of pointer
        callable* self = static_cast<callable*>(ptr);

        // XXX seems like creating a new allocator here is cheating
        //     we should use some member allocator, but it's not clear where to put it
        allocator_type alloc_;
        alloc_.deallocate(self, 1);
      }
    };

    // this deleter calls the destructor of its argument but does not
    // deallocate the ptr
    // T will deallocate itself inside ~T()
    struct self_deallocator_deleter
    {
      template<class T>
      __AGENCY_ANNOTATION
      void operator()(T* ptr) const
      {
        ptr->~T();
      }
    };

    using function_pointer = detail::unique_ptr<callable_self_deallocator_base, self_deallocator_deleter>;

    template<class Alloc, class Function>
    __AGENCY_ANNOTATION
    static function_pointer allocate_function_pointer(const Alloc& alloc, Function&& f)
    {
      using concrete_function_type = callable<typename std::decay<Function>::type, Alloc>;
      return agency::detail::allocate_unique_with_deleter<concrete_function_type>(alloc, self_deallocator_deleter(), alloc, std::forward<Function>(f));
    }

    template<class T>
    struct default_allocator
    {
      using value_type = T;

      default_allocator() = default;

      default_allocator(const default_allocator&) = default;

      template<class U>
      __AGENCY_ANNOTATION
      default_allocator(const default_allocator<U>&) {}

      // XXX we have to implement this member function superfluously because
      //     agency::detail::allocate_unique calls it directly instead of using std::allocator_traits
      template<class U, class... OtherArgs>
      __AGENCY_ANNOTATION
      void construct(U* ptr, OtherArgs&&... args)
      {
        ::new(ptr) U(std::forward<OtherArgs>(args)...);
      }

      value_type* allocate(size_t n)
      {
        std::allocator<T> alloc;
        return alloc.allocate(n);
      }

      void deallocate(value_type* ptr, std::size_t n)
      {
        std::allocator<value_type> alloc;
        alloc.deallocate(ptr, n);
      }
    };


    function_pointer f_ptr_; 
};


} // end detail
} // end agency

