#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/utility.hpp>
#include <agency/experimental/variant.hpp>
#include <agency/experimental/optional.hpp>
#include <agency/future/future_traits/is_future.hpp>
#include <agency/future/future_traits/future_result.hpp>
#include <agency/detail/terminate.hpp>

#include <exception>
#include <type_traits>
#include <future>

namespace agency
{

// declare always_ready_future for detail::always_ready_then below
template<class T>
class always_ready_future;


namespace detail
{


// declare always_ready_then_and_leave() for always_ready_future::then() below
template<class T, class Function>
__AGENCY_ANNOTATION
always_ready_future<detail::result_of_t<Function&&(T&)>>
  always_ready_then_and_leave_valid(always_ready_future<T>& future, Function&& f);

template<class Function>
__AGENCY_ANNOTATION
always_ready_future<detail::result_of_t<Function&&()>>
  always_ready_then_and_leave_valid(always_ready_future<void>& future, Function&& f);


} // end detail


// always_ready_future is a future that is created in a ready state
// this type of future is always ready as long as it is valid
//
// Executors which always block their client can use always_ready_future as their
// associated future and still expose two-way asynchronous execution functions like twoway_execute()

template<class T>
class always_ready_future
{
  public:
    // Default constructor creates an invalid always_ready_future
    // Postcondition: !valid()
    always_ready_future() = default;

    __AGENCY_ANNOTATION
    always_ready_future(const T& value) : state_(value) {}

    __AGENCY_ANNOTATION
    always_ready_future(T&& value) : state_(std::move(value)) {}

    __AGENCY_ANNOTATION
    always_ready_future(std::exception_ptr e) : state_(e) {}

    __AGENCY_ANNOTATION
    always_ready_future(always_ready_future&& other)
      : state_()
    {
      detail::adl_swap(state_, other.state_);
    }

    // converting move constructor waits on the given future and moves its result into *this
    // postcondition: !other.valid()
    __agency_exec_check_disable__
    template<class Future,
             __AGENCY_REQUIRES(is_future<detail::decay_t<Future>>::value),
             __AGENCY_REQUIRES(std::is_constructible<T, future_result_t<detail::decay_t<Future>>&&>::value)
            >
    __AGENCY_ANNOTATION
    always_ready_future(Future&& other)
      : always_ready_future(other.get())
    {
      // XXX this implementation doesn't correctly handle exceptional other
    }

    __AGENCY_ANNOTATION
    always_ready_future& operator=(always_ready_future&& other)
    {
      state_.reset();
      detail::adl_swap(state_, other.state_);
      return *this;
    }

    __agency_exec_check_disable__
    template<class Future,
             __AGENCY_REQUIRES(is_future<detail::decay_t<Future>>::value),
             __AGENCY_REQUIRES(std::is_constructible<T, future_result_t<detail::decay_t<Future>>&&>::value)
            >
    __AGENCY_ANNOTATION
    always_ready_future& operator=(Future&& other)
    {
      return operator=(always_ready_future{std::move(other)});
    }

    template<class U,
             __AGENCY_REQUIRES(
               std::is_constructible<T,U&&>::value
             )>
    __AGENCY_ANNOTATION
    static always_ready_future make_ready(U&& value)
    {
      return always_ready_future(std::forward<U>(value));
    }

    __AGENCY_ANNOTATION
    constexpr static bool is_ready()
    {
      return true;
    }

  private:
    struct get_ptr_visitor
    {
      __AGENCY_ANNOTATION
      T* operator()(T& result) const
      {
        return &result;
      }

      __AGENCY_ANNOTATION
      T* operator()(std::exception_ptr& e) const
      {
#ifndef __CUDA_ARCH__
        std::rethrow_exception(e);
#else
        detail::terminate_with_message("always_ready_future::get_ref(): error: exceptional future");
#endif
        return nullptr;
      }
    };

  public:
    __AGENCY_ANNOTATION
    T& get_ref()
    {
      if(!valid())
      {
#ifndef __CUDA_ARCH__
        throw std::future_error(std::future_errc::no_state);
#else
        detail::terminate_with_message("always_ready_future::get_ref(): error: no state");
#endif
      }

      return *experimental::visit(get_ptr_visitor(), *state_);
    }

    __AGENCY_ANNOTATION
    T get()
    {
      T result = std::move(get_ref());

      invalidate();

      return result;
    }

    __AGENCY_ANNOTATION
    void wait() const
    {
      // wait() is a no-op: *this is always ready
    }

    __AGENCY_ANNOTATION
    bool valid() const
    {
      return state_.has_value();
    }

    template<class Function>
    __AGENCY_ANNOTATION
    always_ready_future<detail::result_of_t<detail::decay_t<Function>(T&)>>
      then_and_leave_valid(Function&& f)
    {
      return detail::always_ready_then_and_leave_valid(*this, std::forward<Function>(f));
    }

    template<class Function>
    __AGENCY_ANNOTATION
    always_ready_future<detail::result_of_t<detail::decay_t<Function>(T&)>>
      then(Function&& f)
    {
      auto result = then_and_leave_valid(std::forward<Function>(f));

      invalidate();

      return result;
    }

  private:
    __AGENCY_ANNOTATION
    void invalidate()
    {
      state_ = experimental::nullopt;
    }

    experimental::optional<experimental::variant<T,std::exception_ptr>> state_;
};


template<>
class always_ready_future<void>
{
  public:
    // XXX the default constructor creates a ready, valid future, but we may wish
    //     to redefine the default constructor to create an invalid future
    //     in such a scheme, we would need to distinguish another constructor for
    //     creating a ready (void) result.
    //     We could create an emplacing constructor distinguished with an in_place_t parameter
    __AGENCY_ANNOTATION
    always_ready_future() : valid_(true) {}

    always_ready_future(std::exception_ptr e) : exception_(e), valid_(true) {}

    __AGENCY_ANNOTATION
    always_ready_future(always_ready_future&& other)
      : exception_(), valid_(false)
    {
      detail::adl_swap(exception_, other.exception_);
      detail::adl_swap(valid_, other.valid_);
    }

    // converting constructor waits on and invalidates the given future
    __agency_exec_check_disable__
    template<class Future,
             __AGENCY_REQUIRES(is_future<detail::decay_t<Future>>::value),
             __AGENCY_REQUIRES(std::is_void<future_result_t<detail::decay_t<Future>>>::value)
            >
    __AGENCY_ANNOTATION
    always_ready_future(Future&& other)
      : always_ready_future()
    {
      // XXX this implementation doesn't correctly handle exceptional other
      other.get();
    }

    __AGENCY_ANNOTATION
    always_ready_future& operator=(always_ready_future&& other)
    {
      exception_.reset();
      valid_ = false;

      detail::adl_swap(exception_, other.exception_);
      detail::adl_swap(valid_, other.valid_);
      return *this;
    }

    // converting assignment operator waits on the given future
    __agency_exec_check_disable__
    template<class Future,
             __AGENCY_REQUIRES(is_future<detail::decay_t<Future>>::value),
             __AGENCY_REQUIRES(std::is_void<future_result_t<detail::decay_t<Future>>>::value)
            >
    __AGENCY_ANNOTATION
    always_ready_future& operator=(Future&& other)
    {
      return operator=(always_ready_future{std::move(other)});
    }

    __AGENCY_ANNOTATION
    constexpr static bool is_ready()
    {
      return true;
    }

    __AGENCY_ANNOTATION
    static always_ready_future make_ready()
    {
      return always_ready_future();
    }

  public:
    __AGENCY_ANNOTATION
    void get()
    {
      if(!valid())
      {
#ifndef __CUDA_ARCH__
        throw std::future_error(std::future_errc::no_state);
#else
        detail::terminate_with_message("always_ready_future::get(): error: no state");
#endif
      }

      invalidate();

      if(exception_)
      {
#ifndef __CUDA_ARCH__
        std::rethrow_exception(exception_.value());
#else
        detail::terminate_with_message("always_ready_future::get(): error: exceptional future");
#endif
      }
    }

    __AGENCY_ANNOTATION
    void wait() const
    {
      // wait() is a no-op: this is always ready
    }

    __AGENCY_ANNOTATION
    bool valid() const
    {
      return valid_;
    }

    template<class Function>
    __AGENCY_ANNOTATION
    always_ready_future<detail::result_of_t<detail::decay_t<Function>()>>
      then_and_leave_valid(Function&& f)
    {
      return detail::always_ready_then_and_leave_valid(*this, std::forward<Function>(f));
    }

    template<class Function>
    __AGENCY_ANNOTATION
    always_ready_future<detail::result_of_t<detail::decay_t<Function>()>>
      then(Function&& f)
    {
      auto result = then_and_leave_valid(std::forward<Function>(f));
      invalidate();
      return result;
    }

  private:
    __AGENCY_ANNOTATION
    void invalidate()
    {
      valid_ = false;
    }

    experimental::optional<std::exception_ptr> exception_;

    bool valid_;
};


template<class T>
__AGENCY_ANNOTATION
always_ready_future<detail::decay_t<T>> make_always_ready_future(T&& value)
{
  return always_ready_future<detail::decay_t<T>>(std::forward<T>(value));
}

__AGENCY_ANNOTATION
inline always_ready_future<void> make_always_ready_future()
{
  return always_ready_future<void>();
}


template<class T>
always_ready_future<T> make_always_ready_exceptional_future(std::exception_ptr e)
{
  return always_ready_future<T>(e);
}


__agency_exec_check_disable__
template<class Function,
         class... Args,
         __AGENCY_REQUIRES(
           !std::is_void<detail::result_of_t<Function&&(Args&&...)>>::value
         )>
__AGENCY_ANNOTATION
always_ready_future<
  detail::result_of_t<Function&&(Args&&...)>
>
try_make_always_ready_future(Function&& f, Args&&... args)
{
#ifndef __CUDA_ARCH__
  try
  {
    return make_always_ready_future(std::forward<Function>(f)(std::forward<Args>(args)...));
  }
  catch(...)
  {
    using result_type = detail::result_of_t<Function&&(Args&&...)>;

    return make_always_ready_exceptional_future<result_type>(std::current_exception());
  }
#else
  return make_always_ready_future(std::forward<Function>(f)(std::forward<Args>(args)...));
#endif
}


__agency_exec_check_disable__
template<class Function,
         class... Args,
         __AGENCY_REQUIRES(
           std::is_void<detail::result_of_t<Function&&(Args&&...)>>::value
         )>
__AGENCY_ANNOTATION
always_ready_future<void> try_make_always_ready_future(Function&& f, Args&&... args)
{
#ifndef __CUDA_ARCH__
  try
  {
    std::forward<Function>(f)(std::forward<Args>(args)...);
    return make_always_ready_future();
  }
  catch(...)
  {
    return make_always_ready_exceptional_future<void>(std::current_exception());
  }
#else
  std::forward<Function>(f)(std::forward<Args>(args)...);
  return make_always_ready_future();
#endif
}


namespace detail
{


template<class T, class Function>
__AGENCY_ANNOTATION
always_ready_future<detail::result_of_t<Function&&(T&)>>
  always_ready_then_and_leave_valid(always_ready_future<T>& future, Function&& f)
{
  T& argument = future.get_ref();
  return agency::try_make_always_ready_future(std::forward<Function>(f), argument);
}


template<class Function>
__AGENCY_ANNOTATION
always_ready_future<detail::result_of_t<Function&&()>>
  always_ready_then_and_leave_valid(always_ready_future<void>& future, Function&& f)
{
  return agency::try_make_always_ready_future(std::forward<Function>(f));
}


} // end detail
} // end agency

