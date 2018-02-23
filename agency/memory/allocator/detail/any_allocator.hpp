#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/memory/allocator/detail/allocator_traits/is_allocator.hpp>
#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <cassert>


namespace agency
{
namespace detail
{


// XXX this is very similar to std::memory_resource
//     consider just implementing that and using it instead
// XXX abstract_allocator is used as an abstract base class for type erasing allocators
struct abstract_allocator
{
  __AGENCY_ANNOTATION
  virtual ~abstract_allocator() {}

  // XXX these member functions below should be pure virtual, but
  //     nvcc has trouble with that
  //     as a workaround, define them

  __AGENCY_ANNOTATION
  //virtual void copy_construct_into(abstract_allocator& to) const = 0;
  virtual void copy_construct_into(abstract_allocator&) const {}

  __AGENCY_ANNOTATION
  //virtual void copy_assign_to(abstract_allocator& to) const = 0;
  virtual void copy_assign_to(abstract_allocator&) const {}

  // virtual const std::type_info& type() const = 0;
  virtual const std::type_info& type() const { return typeid(void); }

  __AGENCY_ANNOTATION
  //virtual void* allocate(std::size_t n) = 0;
  virtual void* allocate(std::size_t) { return nullptr; }

  __AGENCY_ANNOTATION
  //virtual void deallocate(void* ptr, std::size_t n) = 0;
  virtual void deallocate(void*, std::size_t) {}

  __AGENCY_ANNOTATION
  //virtual bool equal_to(const abstract_allocator& other) const = 0;
  virtual bool equal_to(const abstract_allocator&) const { return false; }

  __AGENCY_ANNOTATION
  //virtual bool not_equal_to(const abstract_allocator& other) const = 0;
  virtual bool not_equal_to(const abstract_allocator&) const { return true; }
};


template<class Allocator>
struct concrete_allocator : abstract_allocator
{
  __agency_exec_check_disable__
  concrete_allocator(const concrete_allocator& other) = default;

  __agency_exec_check_disable__
  ~concrete_allocator() = default;

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  concrete_allocator(const Allocator& allocator) : concrete_allocator_(allocator) {}

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  void copy_construct_into(abstract_allocator& to) const
  {
    concrete_allocator& other = static_cast<concrete_allocator&>(to);

    // copy construct into other
    new (&other) concrete_allocator(*this);
  }

  __AGENCY_ANNOTATION
  void copy_assign_to(abstract_allocator& to) const
  {
    concrete_allocator& other = static_cast<concrete_allocator&>(to);

    // copy assign a concrete allocator to other
    other = *this;
  }

  const std::type_info& type() const
  {
    return typeid(Allocator);
  }

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  void* allocate(std::size_t n)
  {
    // call the concrete allocator object
    return concrete_allocator_.allocate(n);
  }

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  void deallocate(void* ptr, std::size_t n)
  {
    // call the concrete allocator object
    return concrete_allocator_.deallocate(reinterpret_cast<char*>(ptr), n);
  }

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  bool equal_to(const abstract_allocator& other) const
  {
    const concrete_allocator& concrete_other = static_cast<const concrete_allocator&>(other);

    // compare == with other
    return concrete_allocator_ == concrete_other.concrete_allocator_;
  }

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  bool not_equal_to(const abstract_allocator& other) const
  {
    const concrete_allocator& concrete_other = static_cast<const concrete_allocator&>(other);

    // compare != with other
    return concrete_allocator_ != concrete_other.concrete_allocator_;
  }

  // the type of the allocator we store is Allocator rebound to allocate raw bytes
  using concrete_allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<char>;

  mutable concrete_allocator_type concrete_allocator_;
};



template<class T>
class any_small_allocator
{
  public:
    static const size_t max_size = 4 * sizeof(void*);

    using value_type = T;

    any_small_allocator()
      : any_small_allocator(std::allocator<T>())
    {}

    template<class U>
    __AGENCY_ANNOTATION
    any_small_allocator(const any_small_allocator<U>& other)
    {
      // call the allocator's copy constructor
      other.get_abstract_allocator().copy_construct_into(get_abstract_allocator());
    }

    __agency_exec_check_disable__
    template<class Allocator,
             __AGENCY_REQUIRES(
               !std::is_same<Allocator,any_small_allocator>::value and
               (sizeof(Allocator) <= max_size) and
               detail::is_allocator<Allocator>::value
             )>
    __AGENCY_ANNOTATION
    any_small_allocator(const Allocator& allocator)
    {
      // rebind Allocator to get an allocator for T
      using rebound_allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<T>;
      rebound_allocator_type rebound_allocator = allocator;

      new (&storage_) concrete_allocator<rebound_allocator_type>(rebound_allocator);
    }

    __AGENCY_ANNOTATION
    ~any_small_allocator()
    {
      // call the allocator's destructor
      get_abstract_allocator().~abstract_allocator();
    }

    any_small_allocator& operator=(const any_small_allocator& other)
    {
      if(type() == other.type())
      {
        // the types match, just call the copy assign function
        other.get_abstract_allocator().copy_assign_to(get_abstract_allocator());
      }
      else
      {
        // the types match, need to destroy and then copy construct

        // destroy our value
        get_abstract_allocator().~abstract_allocator();

        // copy construct from the other value
        other.get_abstract_allocator().copy_construct_into(get_abstract_allocator());
      }

      return *this;
    }

    const std::type_info& type() const
    {
      return get_abstract_allocator().type();
    }

    __AGENCY_ANNOTATION
    T* allocate(std::size_t n)
    {
      // allocate raw bytes using the abstract allocator and reinterpret these bytes into T
      return reinterpret_cast<T*>(get_abstract_allocator().allocate(n * sizeof(T)));
    }

    __AGENCY_ANNOTATION
    void deallocate(T* ptr, std::size_t n)
    {
      // reinterpret ptr into a pointer to raw bytes and deallocate using the abstract allocator
      get_abstract_allocator().deallocate(ptr, n * sizeof(T));
    }

    bool operator==(const any_small_allocator& other) const
    {
      if(type() == other.type())
      {
        // the types match, call equal_to()
        return get_abstract_allocator().equal_to(other.get_abstract_allocator());
      }

      return false;
    }

    bool operator!=(const any_small_allocator& other) const
    {
      if(type() == other.type())
      {
        // the types match, call not_equal_to()
        return get_abstract_allocator().not_equal_to(other.get_abstract_allocator());
      }

      return true;
    }

    template<class Allocator>
    Allocator& get()
    {
      if(type() != typeid(Allocator))
      {
        assert(0);
      }

      concrete_allocator<Allocator>& storage = static_cast<concrete_allocator<Allocator>&>(get_abstract_allocator());

      return storage.concrete_allocator_;
    }

  private:
    // any_small_allocator's constructor needs access to the .get_abstract_allocator() function of all other types of any_small_allocator
    template<class U> friend class any_small_allocator;

    __AGENCY_ANNOTATION
    abstract_allocator& get_abstract_allocator()
    {
      return *reinterpret_cast<abstract_allocator*>(&storage_);
    }

    __AGENCY_ANNOTATION
    const abstract_allocator& get_abstract_allocator() const
    {
      return *reinterpret_cast<const abstract_allocator*>(&storage_);
    }

    // untyped storage for the contained allocator
    typename std::aligned_storage<max_size>::type storage_;
};


} // end detail
} // end agency


