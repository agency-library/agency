#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/optional.hpp>
#include <agency/cuda/detail/boxed_value.hpp>
#include <agency/cuda/detail/memory/unique_ptr.hpp>
#include <agency/detail/unit.hpp>
#include <agency/detail/factory.hpp>
#include <agency/future.hpp>
#include <agency/cuda/detail/future/async_future.hpp>
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
    bool is_ready() const
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
    // calls f on the object contained within this deferred_result
    // undefined is !is_ready()
    template<class Function>
    __AGENCY_ANNOTATION
    auto fmap(Function&& f) ->
      decltype(agency::invoke(std::forward<Function>(f), this->ref()))
    {
      return agency::invoke(std::forward<Function>(f), ref());
    }

    // calls f on the object contained within this deferred_result
    // and invalidates this deferred_result
    // undefined is !is_ready()
    template<class Function>
    __AGENCY_ANNOTATION
    auto fmap_and_invalidate(Function&& f) ->
      decltype(this->fmap(std::forward<Function>(f)))
    {
      invalidate_at_scope_exit invalidator{*this};
      return fmap(std::forward<Function>(f));
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
    bool is_ready() const
    {
      return static_cast<bool>(this->value());
    }

    __AGENCY_ANNOTATION
    void ref() const
    {
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
    // calls f
    // undefined is !is_ready()
    template<class Function>
    __AGENCY_ANNOTATION
    auto fmap(Function&& f) ->
      decltype(agency::invoke(std::forward<Function>(f)))
    {
      return agency::invoke(std::forward<Function>(f));
    }

    // calls f and invalidates this deferred_result
    // undefined is !is_ready()
    template<class Function>
    __AGENCY_ANNOTATION
    auto fmap_and_invalidate(Function&& f) ->
      decltype(this->fmap(std::forward<Function>(f)))
    {
      invalidate_at_scope_exit invalidator{*this};
      return fmap(std::forward<Function>(f));
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
      return function_ || result_.is_ready();
    }

    __AGENCY_ANNOTATION
    bool is_ready() const
    {
      return valid() && result_.is_ready();
    }

    __AGENCY_ANNOTATION
    void wait()
    {
      if(!is_ready())
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
    // waits for the state to become ready and calls f on the object contained within the state, if any
    // invalidates the state
    template<class Function>
    __AGENCY_ANNOTATION
    auto fmap(Function&& f) ->
      decltype(result_.fmap_and_invalidate(std::forward<Function>(f)))
    {
      wait();
      return result_.fmap_and_invalidate(std::forward<Function>(f));
    }

    // waits for the state to become ready and calls f on the object contained within the state, if any
    // leaves the state valid
    template<class Function>
    __AGENCY_ANNOTATION
    auto fmap_and_leave_valid(Function&& f) ->
      decltype(result_.fmap(std::forward<Function>(f)))
    {
      wait();
      return result_.fmap(std::forward<Function>(f));
    }

    // waits for the state to become ready and move-constructs the object contained within the state, if any
    // invalidates the state
    __AGENCY_ANNOTATION
    T get()
    {
      return fmap(move_functor());
    }

  private:
    struct get_ref_functor
    {
      template<class U>
      __AGENCY_ANNOTATION
      U& operator()(U& arg) const
      {
        return arg;
      }

      __AGENCY_ANNOTATION
      void operator()() const
      {
      }
    };

  public:
    // waits for the state to become ready and returns a reference to the object contained within the state, if any
    // leaves the state valid
    __AGENCY_ANNOTATION
    auto get_ref() ->
      decltype(this->fmap_and_leave_valid(get_ref_functor()))
    {
      return fmap_and_leave_valid(get_ref_functor());
    }
};


// XXX why is this functor external nto deferred_future?
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
    bool is_ready() const
    {
      return state_.is_ready();
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

  // XXX leave this public so uber_future can use it
  //     make it private once uber_future's functionality has been fully integrated into the codebase
  public:
    __AGENCY_ANNOTATION
    auto get_ref() ->
      decltype(state_.get_ref())
    {
      return state_.get_ref();
    }

  private:
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

    template<class Function, class Factory, class Shape, class IndexFunction, class OuterFactory, class InnerFactory>
    struct bulk_then_functor
    {
      Function f;
      Factory result_factory;
      Shape shape;
      IndexFunction index_function;
      OuterFactory outer_factory;
      InnerFactory inner_factory;
      device_id device;

      // operator() for non-void past_arg
      template<class U>
      __host__ __device__
      typename std::result_of<Factory(Shape)>::type
        operator()(U& past_arg)
      {
        auto ready = async_future<U>::make_ready(std::move(past_arg));

        return ready.bulk_then(f, result_factory, shape, index_function, outer_factory, inner_factory, device).get();
      }

      // operator() for void past_arg
      __host__ __device__
      typename std::result_of<Factory(Shape)>::type
        operator()()
      {
        auto ready = async_future<void>::make_ready();

        return ready.bulk_then(f, result_factory, shape, index_function, outer_factory, inner_factory, device).get();
      }
    };

    template<class Function, class Factory, class Shape, class IndexFunction, class OuterFactory, class InnerFactory>
    __host__ __device__
    deferred_future<typename std::result_of<Factory(Shape)>::type>
      bulk_then(Function f, Factory result_factory, Shape shape, IndexFunction index_function, OuterFactory outer_factory, InnerFactory inner_factory, device_id device)
    {
      bulk_then_functor<Function,Factory,Shape,IndexFunction,OuterFactory,InnerFactory> continuation{f,result_factory,shape,index_function,outer_factory,inner_factory,device};

      return then(continuation);
    }

    template<class Function>
    struct then_and_leave_valid_functor
    {
      mutable Function f_;
      deferred_future& predecessor_;
    
      __AGENCY_ANNOTATION
      auto operator()() const ->
        decltype(predecessor_.state_.fmap_and_leave_valid(f_))
      {
        return predecessor_.state_.fmap_and_leave_valid(f_);
      }
    };

  // XXX leave these public so uber_future can use them
  //     make them private once uber_future's functionality has been fully integrated into the codebase
  public:
    template<class Function>
    __AGENCY_ANNOTATION
    deferred_future<
      agency::detail::result_of_continuation_t<
        typename std::decay<Function>::type,
        deferred_future
      >
    >
      then_and_leave_valid(Function&& f)
    {
      auto continuation = then_and_leave_valid_functor<typename std::decay<Function>::type>{std::forward<Function>(f), *this};
      using result_type = agency::detail::result_of_continuation_t<typename std::decay<Function>::type, deferred_future>;
      return deferred_future<result_type>(std::move(continuation));
    }

    template<class Function, class Factory, class Shape, class IndexFunction, class OuterFactory, class InnerFactory>
    __host__ __device__
    deferred_future<typename std::result_of<Factory(Shape)>::type>
      bulk_then_and_leave_valid(Function f, Factory result_factory, Shape shape, IndexFunction index_function, OuterFactory outer_factory, InnerFactory inner_factory, device_id device)
    {
      printf("deferred_future::bulk_then_and_leave_valid(): Unimplemented.\n");

      return deferred_future<typename std::result_of<Factory(Shape)>::type>();
    }
};

} // end cuda
} // end agency

