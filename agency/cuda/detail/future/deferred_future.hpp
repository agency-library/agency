#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/optional.hpp>
#include <agency/cuda/detail/boxed_value.hpp>
#include <agency/cuda/detail/memory/unique_ptr.hpp>
#include <agency/detail/unit.hpp>
#include <agency/detail/factory.hpp>
#include <agency/future.hpp>
#include <agency/cuda/future.hpp>
#include <agency/functional.hpp>


namespace agency
{
namespace cuda
{
namespace detail
{


template<class>
class unique_function;

template<class Result, class... Args>
class unique_function<Result(Args...)>
{
  public:
    __AGENCY_ANNOTATION
    unique_function() = default;

    __AGENCY_ANNOTATION
    unique_function(unique_function&& other) = default;

    template<class Function>
    __AGENCY_ANNOTATION
    unique_function(Function&& f)
      : f_ptr_(make_function_pointer(std::forward<Function>(f)))
    {}

    __AGENCY_ANNOTATION
    unique_function& operator=(unique_function&& other) = default;

    __AGENCY_ANNOTATION
    Result operator()(Args... args) const
    {
      return (*f_ptr_)(args...);
    }

    __AGENCY_ANNOTATION
    operator bool () const
    {
      return f_ptr_;
    }

  private:
    struct callable_base
    {
      __AGENCY_ANNOTATION
      virtual ~callable_base() {}

      __AGENCY_ANNOTATION
      virtual Result operator()(Args... args) const = 0;
    };

    template<class Function>
    struct callable : callable_base
    {
      mutable Function f_;

      template<class OtherFunction,
               class = typename std::enable_if<
                 std::is_constructible<Function,OtherFunction&&>::value
               >::type>
      __AGENCY_ANNOTATION
      callable(OtherFunction&& f)
        : f_(std::forward<OtherFunction>(f))
      {}

      __AGENCY_ANNOTATION
      virtual Result operator()(Args... args) const
      {
        return f_(args...);
      }
    };

    using function_pointer = detail::unique_ptr<callable_base>;

    template<class Function>
    __AGENCY_ANNOTATION
    static function_pointer make_function_pointer(Function&& f)
    {
      using concrete_function_type = callable<typename std::decay<Function>::type>;
      return detail::make_unique<concrete_function_type>(std::forward<Function>(f));
    }

    function_pointer f_ptr_; 
};


template<class T>
class deferred_result : detail::boxed_value<agency::detail::optional<T>>
{
  public:
    using super_t = detail::boxed_value<agency::detail::optional<T>>;
    using super_t::super_t;
    using super_t::operator=;

    __AGENCY_ANNOTATION
    deferred_result(deferred_result&& other)
      : super_t(std::move(other))
    {
      // empty other
      other.value() = agency::detail::nullopt;
    }

    __AGENCY_ANNOTATION
    deferred_result& operator=(deferred_result&& other)
    {
      super_t::operator=(std::move(other));

      // empty other
      other.value() = agency::detail::nullopt;

      return *this;
    }

    __AGENCY_ANNOTATION
    bool ready() const
    {
      return static_cast<bool>(this->value());
    }

    __AGENCY_ANNOTATION
    T& ref()
    {
      return this->value().value();
    }

  private:
    struct invalidate_at_scope_exit
    {
      deferred_result& self;

      __AGENCY_ANNOTATION
      ~invalidate_at_scope_exit()
      {
        self.value() = agency::detail::nullopt;
      }
    };

  public:
    template<class Function>
    __AGENCY_ANNOTATION
    auto fmap(Function&& f) ->
      decltype(agency::invoke(std::forward<Function>(f), this->ref()))
    {
      invalidate_at_scope_exit invalidator{*this};
      return agency::invoke(std::forward<Function>(f), ref());
    }
};

template<>
class deferred_result<void> : detail::boxed_value<agency::detail::optional<agency::detail::unit>>
{
  public:
    using super_t = detail::boxed_value<agency::detail::optional<agency::detail::unit>>;
    using super_t::super_t;
    using super_t::operator=;

    __AGENCY_ANNOTATION
    deferred_result(deferred_result&& other)
      : super_t(std::move(other))
    {
      // empty other
      other.value() = agency::detail::nullopt;
    }

    __AGENCY_ANNOTATION
    deferred_result& operator=(deferred_result&& other)
    {
      super_t::operator=(std::move(other));

      // empty other
      other.value() = agency::detail::nullopt;

      return *this;
    }

    __AGENCY_ANNOTATION
    bool ready() const
    {
      return static_cast<bool>(this->value());
    }

  private:
    struct invalidate_at_scope_exit
    {
      deferred_result& self;

      __AGENCY_ANNOTATION
      ~invalidate_at_scope_exit()
      {
        self.value() = agency::detail::nullopt;
      }
    };

  public:
    template<class Function>
    __AGENCY_ANNOTATION
    auto fmap(Function&& f) ->
      decltype(agency::invoke(std::forward<Function>(f)))
    {
      invalidate_at_scope_exit invalidator{*this};
      return agency::invoke(std::forward<Function>(f));
    }
};


template<class T>
class deferred_state
{
  private:
    deferred_result<T> result_;
    unique_function<T()> function_;

    template<class U,
             class = typename std::enable_if<
               std::is_void<U>::value
             >::type>
    __AGENCY_ANNOTATION
    agency::detail::unit invoke_function()
    {
      function_();
      return agency::detail::unit();
    }

    template<class U,
             class = typename std::enable_if<
               !std::is_void<U>::value
             >::type>
    __AGENCY_ANNOTATION
    T invoke_function()
    {
      return function_();
    }

  public:
    __AGENCY_ANNOTATION
    deferred_state() = default;

    __AGENCY_ANNOTATION
    deferred_state(deferred_state&&) = default;

    template<class Function>
    __AGENCY_ANNOTATION
    deferred_state(Function&& f)
      : function_(std::forward<Function>(f))
    {}

    __AGENCY_ANNOTATION
    deferred_state& operator=(deferred_state&& other) = default;

    __AGENCY_ANNOTATION
    bool valid() const
    {
      return function_ || result_.ready();
    }

    __AGENCY_ANNOTATION
    bool ready() const
    {
      return valid() && result_.ready();
    }

    __AGENCY_ANNOTATION
    void wait()
    {
      if(!ready())
      {
        result_ = invoke_function<T>();

        // empty the function
        function_ = unique_function<T()>();
      }
    }

  private:
    struct move_functor
    {
      template<class U>
      __AGENCY_ANNOTATION
      U operator()(U& arg) const
      {
        return std::move(arg); 
      }

      __AGENCY_ANNOTATION
      void operator()() const {}
    };

  public:
    template<class Function>
    __AGENCY_ANNOTATION
    auto fmap(Function&& f) ->
      decltype(result_.fmap(std::forward<Function>(f)))
    {
      wait();
      return result_.fmap(std::forward<Function>(f));
    }

    __AGENCY_ANNOTATION
    T get()
    {
      return fmap(move_functor());
    }
};


template<class Function, class DeferredFuture>
struct deferred_continuation
{
  mutable Function f_;
  mutable DeferredFuture predecessor_;

  __AGENCY_ANNOTATION
  auto operator()() const ->
    decltype(predecessor_.fmap(f_))
  {
    return predecessor_.fmap(f_);
  }
};


} // end detail


template<class T>
class deferred_future
{
  public:
    __AGENCY_ANNOTATION
    deferred_future() = default;

    __AGENCY_ANNOTATION
    deferred_future(deferred_future&&) = default;

    __AGENCY_ANNOTATION
    deferred_future& operator=(deferred_future&& other) = default;

    __AGENCY_ANNOTATION
    bool valid() const
    {
      return state_.valid();
    }

    __AGENCY_ANNOTATION
    bool ready() const
    {
      return state_.ready();
    }

    __AGENCY_ANNOTATION
    void wait()
    {
      state_.wait();
    }

    __AGENCY_ANNOTATION
    T get()
    {
      return state_.get();
    }

    template<class... Args,
             class = typename std::enable_if<
               detail::is_constructible_or_void<T,Args...>::value
             >::type>
    __AGENCY_ANNOTATION
    static deferred_future make_ready(Args&&... args)
    {
      deferred_future result(agency::detail::make_factory<T>(std::forward<Args>(args)...));
      result.wait();
      return std::move(result);
    }

    template<class Function>
    __AGENCY_ANNOTATION
    deferred_future<
      agency::detail::result_of_continuation_t<
        typename std::decay<Function>::type,
        deferred_future
      >
    >
      then(Function&& f)
    {
      auto continuation = detail::deferred_continuation<typename std::decay<Function>::type, deferred_future>{std::forward<Function>(f), std::move(*this)};
      using result_type = agency::detail::result_of_continuation_t<typename std::decay<Function>::type, deferred_future>;
      return deferred_future<result_type>(std::move(continuation));
    }

  private:
    detail::deferred_state<T> state_;

    template<class>
    friend class deferred_future;

    template<class,class>
    friend struct agency::cuda::detail::deferred_continuation;

    template<class Function,
             class = typename std::enable_if<
               std::is_constructible<
                 detail::deferred_state<T>,
                 Function&&
               >::value
             >::type>
    __AGENCY_ANNOTATION
    deferred_future(Function&& f)
      : state_(std::forward<Function>(f))
    {}

    template<class Function>
    __AGENCY_ANNOTATION
    auto fmap(Function&& f) ->
      decltype(state_.fmap(std::forward<Function>(f)))
    {
      return state_.fmap(std::forward<Function>(f));
    }
};

} // end cuda
} // end agency

