#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <iostream>


struct inline_executor
{
  template<class F>
  void execute(F&& f) const
  {
    std::forward<F>(f)();
  }

  friend constexpr bool operator==(const inline_executor&, const inline_executor&)
  {
    return true;
  }

  friend constexpr bool operator!=(const inline_executor&, const inline_executor&)
  {
    return false;
  }
};


int main()
{
  using namespace agency;

  {
    // test query()
    
    static_assert(agency::query(inline_executor(), bulk_guarantee_t())     == bulk_guarantee_t::unsequenced_t(), "Unsequenced is not guaranteed.");
    static_assert(agency::query(vector_executor(), bulk_guarantee_t())     == bulk_guarantee_t::unsequenced_t(), "Unsequenced is not guaranteed.");
    static_assert(agency::query(sequenced_executor(), bulk_guarantee_t())  == bulk_guarantee_t::sequenced_t(),   "Concurrent is not guaranteed.");
    static_assert(agency::query(concurrent_executor(), bulk_guarantee_t()) == bulk_guarantee_t::concurrent_t(),  "Concurrent is not guaranteed.");
    static_assert(agency::query(unsequenced_executor(), bulk_guarantee_t()) == bulk_guarantee_t::unsequenced_t(),  "Unsequenced is not guaranteed.");

    // test parallel_executor with a named variable because for some reason parallel_executor() cannot be used in the static_assert below
    parallel_executor par_ex;
    static_assert(agency::query(par_ex, bulk_guarantee_t()) == bulk_guarantee_t::parallel_t(),  "Parallel is not guaranteed.");

    // CUDA-specific executors
    // XXX test some of these with assert because nvcc falsely believes these bulk_guarantee_t::scoped_t comparisons are not constant expressions
    assert((agency::query(cuda::grid_executor(), bulk_guarantee_t()) == bulk_guarantee_t::scoped_t<bulk_guarantee_t::parallel_t, bulk_guarantee_t::concurrent_t>()));
    assert((agency::query(cuda::concurrent_grid_executor(), bulk_guarantee_t()) == bulk_guarantee_t::scoped_t<bulk_guarantee_t::concurrent_t, bulk_guarantee_t::concurrent_t>()));
    static_assert(agency::query(cuda::concurrent_executor(), bulk_guarantee_t()) == bulk_guarantee_t::concurrent_t(), "Concurrent is not guaranteed.");
    static_assert(agency::query(cuda::parallel_executor(), bulk_guarantee_t()) == bulk_guarantee_t::parallel_t(), "Parallel is not guaranteed.");
  }

  {
    // sequenced -> sequenced
    auto ex = agency::require(sequenced_executor(), bulk_guarantee_t::sequenced_t());

    static_assert(query(ex, bulk_guarantee) == bulk_guarantee_t::sequenced_t(), "Sequenced is not guaranteed.");
    static_assert(std::is_same<sequenced_executor, decltype(ex)>::value, "Result is not the same type as the original.");
  }

  {
    // sequenced -> parallel

    auto ex = agency::require(sequenced_executor(), bulk_guarantee_t::parallel_t());

    static_assert(query(ex, bulk_guarantee) == bulk_guarantee_t::parallel_t(), "Parallel is not guaranteed.");
  }

  {
    // sequenced -> unsequenced

    auto ex = agency::require(sequenced_executor(), bulk_guarantee_t::unsequenced_t());

    static_assert(query(ex, bulk_guarantee) == bulk_guarantee_t::unsequenced_t(), "Unsequenced is not guaranteed.");
  }


  {
    // concurrent -> concurrent
    auto ex = agency::require(concurrent_executor(), bulk_guarantee_t::concurrent_t());

    static_assert(query(ex, bulk_guarantee) == bulk_guarantee_t::concurrent_t(), "Concurrent is not guaranteed.");
    static_assert(std::is_same<concurrent_executor, decltype(ex)>::value, "Result is not the same type as original.");
  }


  {
    // concurrent -> parallel
    auto ex = agency::require(concurrent_executor(), bulk_guarantee_t::parallel_t());

    static_assert(query(ex, bulk_guarantee) == bulk_guarantee_t::parallel_t(), "Parallel is not guaranteed.");
  }


  {
    // concurrent -> unsequenced
    auto ex = agency::require(concurrent_executor(), bulk_guarantee_t::unsequenced_t());

    static_assert(query(ex, bulk_guarantee) == bulk_guarantee_t::unsequenced_t(), "Unsequenced is not guaranteed.");
  }
    
    
  {
    // parallel -> parallel

    auto ex = agency::require(parallel_executor(), bulk_guarantee_t::parallel_t());

    static_assert(query(ex, bulk_guarantee) == bulk_guarantee_t::parallel_t(), "Parallel is not guaranteed.");
    static_assert(std::is_same<parallel_executor, decltype(ex)>::value, "Result is not the same type as original.");
  }

  {
    // parallel -> unsequenced

    auto ex = agency::require(parallel_executor(), bulk_guarantee_t::unsequenced_t());

    static_assert(query(ex, bulk_guarantee) == bulk_guarantee_t::unsequenced_t(), "Unsequenced is not guaranteed.");
  }


  {
    // unsequenced -> unsequenced
    auto ex = agency::require(unsequenced_executor(), bulk_guarantee_t::unsequenced_t());

    static_assert(query(ex, bulk_guarantee) == bulk_guarantee_t::unsequenced_t(), "Unsequenced is not guaranteed.");
    static_assert(std::is_same<unsequenced_executor, decltype(ex)>::value, "Result is not the same type as original.");
  }

  std::cout << "OK" << std::endl;

  return 0;
}

