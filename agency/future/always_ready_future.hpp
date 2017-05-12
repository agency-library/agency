#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/utility.hpp>
#include <agency/experimental/variant.hpp>
#include <agency/experimental/optional.hpp>

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


// declare always_ready_then() for always_ready_future::then() below
template<class T, class Function>
always_ready_future<detail::result_of_t<Function&&(T&)>>
  always_ready_then(always_ready_future<T>& future, Function&& f);

template<class Function>
always_ready_future<detail::result_of_t<Function&&()>>
  always_ready_then(always_ready_future<void>& future, Function&& f);


} // end detail


// always_ready_future is a future that is always created in a ready state
//
// Executors which always block their client can use always_ready_future as their
// associated future and still expose two-way asynchronous execution functions like async_execute()

template<class T>
class always_ready_future
{
  public:
    always_ready_future(const T& value) : state_(value) {}

    always_ready_future(T&& value) : state_(std::move(value)) {}

    always_ready_future(std::exception_ptr e) : state_(e) {}

    always_ready_future(always_ready_future&& other)
      : state_()
    {
      detail::adl_swap(state_, other.state_);
    }

    always_ready_future& operator=(always_ready_future&& other)
    {
      state_.reset();
      detail::adl_swap(state_, other.state_);
      return *this;
    }

    template<class U,
             __AGENCY_REQUIRES(
               std::is_constructible<T,U&&>::value
             )>
    static always_ready_future make_ready(U&& value)
    {
      return always_ready_future(std::forward<U>(value));
    }

  private:
    struct get_visitor
    {
      T operator()(T& value) const
      {
        return std::move(value);
      }

      T operator()(std::exception_ptr& e) const
      {
        throw e;

        // XXX rework this visitor to avoid returning T in both cases
        return T();
      }
    };

  public:
    T get()
    {
      if(!valid())
      {
        throw std::future_error(std::future_errc::no_state);
      }

      T result = experimental::visit(get_visitor(), *state_);

      state_ = experimental::nullopt;

      return result;
    }

    void wait() const
    {
      // wait() is a no-op: this is always ready
    }

    bool valid() const
    {
      return state_.has_value();
    }

    template<class Function>
    always_ready_future<detail::result_of_t<detail::decay_t<Function>(T&)>>
      then(Function&& f)
    {
      return detail::always_ready_then(*this, std::forward<Function>(f));
    }

  private:
    experimental::optional<experimental::variant<T,std::exception_ptr>> state_;
};


template<>
class always_ready_future<void>
{
  public:
    always_ready_future() : valid_(true) {}

    always_ready_future(std::exception_ptr e) : exception_(e), valid_(true) {}

    always_ready_future(always_ready_future&& other)
      : exception_(), valid_(false)
    {
      detail::adl_swap(exception_, other.exception_);
      detail::adl_swap(valid_, other.valid_);
    }

    always_ready_future& operator=(always_ready_future&& other)
    {
      exception_.reset();
      valid_ = false;

      detail::adl_swap(exception_, other.exception_);
      detail::adl_swap(valid_, other.valid_);
      return *this;
    }

    static always_ready_future make_ready()
    {
      return always_ready_future();
    }

  public:
    void get()
    {
      if(!valid())
      {
        throw std::future_error(std::future_errc::no_state);
      }

      valid_ = false;

      if(exception_)
      {
        throw exception_.value();
      }
    }

    void wait() const
    {
      // wait() is a no-op: this is always ready
    }

    bool valid() const
    {
      return valid_;
    }

    template<class Function>
    always_ready_future<detail::result_of_t<detail::decay_t<Function>()>>
      then(Function&& f)
    {
      return detail::always_ready_then(*this, std::forward<Function>(f));
    }

  private:
    experimental::optional<std::exception_ptr> exception_;

    bool valid_;
};


template<class T>
always_ready_future<detail::decay_t<T>> make_always_ready_future(T&& value)
{
  return always_ready_future<detail::decay_t<T>>(std::forward<T>(value));
}

inline always_ready_future<void> make_always_ready_future()
{
  return always_ready_future<void>();
}


template<class T>
always_ready_future<T> make_always_ready_exceptional_future(std::exception_ptr e)
{
  return always_ready_future<T>(e);
}


template<class Function,
         class... Args,
         __AGENCY_REQUIRES(
           !std::is_void<detail::result_of_t<Function&&(Args&&...)>>::value
         )>
always_ready_future<
  detail::result_of_t<Function&&(Args&&...)>
>
try_make_always_ready_future(Function&& f, Args&&... args)
{
  try
  {
    return make_always_ready_future(std::forward<Function>(f)(std::forward<Args>(args)...));
  }
  catch(...)
  {
    using result_type = detail::result_of_t<Function&&(Args&&...)>;

    return make_always_ready_exceptional_future<result_type>(std::current_exception());
  }
}


template<class Function,
         class... Args,
         __AGENCY_REQUIRES(
           std::is_void<detail::result_of_t<Function&&(Args&&...)>>::value
         )>
always_ready_future<void> try_make_always_ready_future(Function&& f, Args&&... args)
{
  try
  {
    std::forward<Function>(f)(std::forward<Args>(args)...);
    return make_always_ready_future();
  }
  catch(...)
  {
    return make_always_ready_exceptional_future<void>(std::current_exception());
  }
}


namespace detail
{


template<class T, class Function>
always_ready_future<detail::result_of_t<Function&&(T&)>>
  always_ready_then(always_ready_future<T>& future, Function&& f)
{
  auto argument = future.get();
  return agency::try_make_always_ready_future(std::forward<Function>(f), argument);
}


template<class Function>
always_ready_future<detail::result_of_t<Function&&()>>
  always_ready_then(always_ready_future<void>& future, Function&& f)
{
  future.get();
  return agency::try_make_always_ready_future(std::forward<Function>(f));
}


} // end detail
} // end agency

