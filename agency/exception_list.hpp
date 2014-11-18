#pragma once

#include <exception>
#include <vector>
#include <iterator>

namespace agency
{


class exception_list;


namespace detail
{


inline void add_current_exception(exception_list &exceptions);
inline void move_exceptions(exception_list &to, exception_list &from);


} // end detail


class exception_list : public std::exception
{
  private:
    typedef std::vector<std::exception_ptr> impl_type;
    impl_type exceptions_;

    friend void detail::add_current_exception(exception_list &);
    friend void detail::move_exceptions(exception_list &, exception_list &);

  public:
    typedef std::exception_ptr                                        value_type;
    typedef const value_type&                                         reference;
    typedef const value_type&                                         const_reference;
    typedef typename impl_type::const_iterator                        const_iterator;
    typedef const_iterator                                            iterator;
    // XXX WAR nvcc 6.5 issue
    //typedef typename iterator_traits<const_iterator>::difference_type difference_type;
    typedef std::ptrdiff_t                                            difference_type;
    typedef size_t                                                    size_type;

    size_t size() const { return exceptions_.size(); }
    iterator begin() const { return exceptions_.begin(); }
    iterator end() const { return exceptions_.end(); }
    inline virtual const char* what() const noexcept { return "exception_list"; }
};


namespace detail
{


inline void add_current_exception(exception_list &exceptions)
{
  exceptions.exceptions_.push_back(std::current_exception());
}


inline void move_exceptions(exception_list &to, exception_list &from)
{
  to.exceptions_.insert(to.exceptions_.end(), std::make_move_iterator(from.exceptions_.begin()), std::make_move_iterator(from.exceptions_.end()));
}


template<class FutureIterator>
exception_list flatten_into_exception_list(FutureIterator first, FutureIterator last)
{
  exception_list exceptions;

  for(; first != last; ++first)
  {
    try
    {
      first->get();
    }
    catch(exception_list &e)
    {
      move_exceptions(exceptions, e);
    }
    catch(...)
    {
      add_current_exception(exceptions);
    }
  }
  
  return exceptions;
}


template<class FutureIterator>
void flatten_and_throw_exceptions(FutureIterator first, FutureIterator last)
{
  exception_list exceptions = flatten_into_exception_list(first, last);

  if(exceptions.size() > 0)
  {
    throw exceptions;
  }
}


} // end detail
} // end agency

