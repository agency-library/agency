#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/memory/unique_ptr.hpp>
#include <agency/detail/is_call_possible.hpp>
#include <type_traits>
#include <typeinfo>
#include <utility>


namespace agency
{
namespace detail
{


template<class T>
class any_small_deleter
{
  public:
    static const size_t max_size = 4 * sizeof(void*);

    using pointer = T*;

    __AGENCY_ANNOTATION
    any_small_deleter()
      : any_small_deleter(default_delete<T>())
    {}

    __AGENCY_ANNOTATION
    any_small_deleter(const any_small_deleter& other)
    {
      // call the deleter's copy constructor
      other.get_abstract_deleter().copy_construct_into(get_abstract_deleter());
    }

    template<class Deleter,
             __AGENCY_REQUIRES(
               !std::is_same<Deleter,any_small_deleter>::value and
               (sizeof(Deleter) <= max_size) and
               is_call_possible<Deleter, pointer>::value
             )>
    __AGENCY_ANNOTATION
    any_small_deleter(const Deleter& d)
    {
      new (&storage_) concrete_deleter<Deleter>(d);
    }

    __AGENCY_ANNOTATION
    ~any_small_deleter()
    {
      // call the deleter's destructor
      get_abstract_deleter().~abstract_deleter();
    }

    any_small_deleter& operator=(const any_small_deleter& other)
    {
      if(type() == other.type())
      {
        // the types match, just call the copy assign function
        other.get_abstract_deleter().copy_assign_to(get_abstract_deleter());
      }
      else
      {
        // the types match, need to destroy and then copy construct

        // destroy our value
        get_abstract_deleter().~abstract_deleter();

        // copy construct from the other value
        other.get_abstract_deleter().copy_construct_into(get_abstract_deleter());
      }

      return *this;
    }

    const std::type_info& type() const
    {
      return get_abstract_deleter().type();
    }

    __AGENCY_ANNOTATION
    void operator()(pointer ptr) const
    {
      get_abstract_deleter()(ptr);
    }

  private:
    struct abstract_deleter
    {
      __AGENCY_ANNOTATION
      virtual ~abstract_deleter() {}

      // XXX these member functions below should be pure virtual, but
      //     nvcc has trouble with that
      //     as a workaround, define them

      __AGENCY_ANNOTATION
      //virtual void copy_construct_into(abstract_deleter& to) const = 0;
      virtual void copy_construct_into(abstract_deleter& to) const {}

      __AGENCY_ANNOTATION
      //virtual void copy_assign_to(abstract_deleter& to) const = 0;
      virtual void copy_assign_to(abstract_deleter& to) const {}

      // virtual const std::type_info& type() const = 0;
      virtual const std::type_info& type() const { return typeid(void); }

      __AGENCY_ANNOTATION
      //virtual void operator()(pointer ptr) const = 0;
      virtual void operator()(pointer ptr) const {}
    };

    template<class Deleter>
    struct concrete_deleter : abstract_deleter
    {
      __AGENCY_ANNOTATION
      concrete_deleter(const Deleter& d) : concrete_deleter_(d) {}

      __AGENCY_ANNOTATION
      void copy_construct_into(abstract_deleter& to) const
      {
        concrete_deleter& other = static_cast<concrete_deleter&>(to);

        // copy construct into other
        new (&other) concrete_deleter(*this);
      }

      __AGENCY_ANNOTATION
      void copy_assign_to(abstract_deleter& to) const
      {
        concrete_deleter& other = static_cast<concrete_deleter&>(to);

        // copy assign a concrete deleter to other
        other = *this;
      }

      const std::type_info& type() const
      {
        return typeid(Deleter);
      }

      __AGENCY_ANNOTATION
      void operator()(pointer ptr) const
      {
        // call the concrete deleter object
        concrete_deleter_(ptr);
      }

      mutable Deleter concrete_deleter_;
    };

    __AGENCY_ANNOTATION
    abstract_deleter& get_abstract_deleter()
    {
      return *reinterpret_cast<abstract_deleter*>(&storage_);
    }

    __AGENCY_ANNOTATION
    const abstract_deleter& get_abstract_deleter() const
    {
      return *reinterpret_cast<const abstract_deleter*>(&storage_);
    }

    // untyped storage for the contained deleter
    typename std::aligned_storage<max_size>::type storage_;
};


} // end detail
} // end agency

