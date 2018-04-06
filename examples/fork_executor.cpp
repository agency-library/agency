#include <agency/agency.hpp>
#include <agency/experimental.hpp>

#include <numeric>
#include <vector>

#include <unistd.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/wait.h>


// This example program demonstrates how to create a user-defined executor which forks a new process every time it creates
// an execution agent.
//
// There are two major components:
//
//   1. The executor itself, which implements its .bulk_sync_execute() function via fork() and
//   2. a special type of allocator for allocating shared memory via mmap() through which the forked processes may communicate.
//
// Finally, we validate that our executor is correct by using it to create execution for a parallel sum algorithm.


// Forked processes may communicate through shared memory which has been dynamically-allocated by mmap.
template<class T>
class shared_memory_allocator
{
  public:
    using value_type = T;

    shared_memory_allocator() = default;

    template <class U>
    shared_memory_allocator(const shared_memory_allocator<U>&) {}

    // allocate calls mmap with the appropriate flags for shared memory
    T* allocate(std::size_t n)
    {
      if(n <= std::numeric_limits<std::size_t>::max() / sizeof(T))
      {
        return static_cast<T*>(mmap(NULL, n * sizeof (T),
                                    PROT_READ | PROT_WRITE,
                                    MAP_SHARED | MAP_ANONYMOUS,
                                    -1, 0));
      }
      throw std::bad_alloc();
    }

    // deallocate just calls munmap
    void deallocate(T* ptr, std::size_t)
    {
      munmap(ptr, sizeof(*ptr));
    }
};


// This executor creates execution by forking a process for each execution agent it creates.
class fork_executor
{
  public:
    // forked processes execute in parallel
    constexpr static agency::bulk_guarantee_t::parallel_t query(agency::bulk_guarantee_t)
    {
      return agency::bulk_guarantee_t::parallel_t();
    }

    // forked processes communicate through shared memory
    template<typename T> using allocator = shared_memory_allocator<T>;

    // the futures returned by this executor are always ready
    template<class T>
    using future = agency::always_ready_future<T>;

    template<class Function, class ResultFactory, class SharedFactory>
    agency::always_ready_future<typename std::result_of<ResultFactory()>::type>
    bulk_twoway_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory) const
    {
      // name the types of the objects returned by the factories
      using result_type = typename std::result_of<ResultFactory()>::type;
      using shared_parm_type = typename std::result_of<SharedFactory()>::type;

      // when each forked child process invokes f, it needs to pass along the objects returned by the two factories
      // because there is only a single object for each ot
      //
      // use our special allocator to create single element containers for the results of the factories
      // XXX these should be unique_ptr, but no allocate_unique() function exists
      std::vector<result_type, allocator<result_type>> result(1, result_factory());
      std::vector<shared_parm_type, allocator<shared_parm_type>> shared_parm(1, shared_factory());

      // create n children with fork
      for(size_t i = 0; i < n; ++i)
      {
        if(fork() == 0)
        {
          // each child invokes f with the result and shared parameter
          f(i, result[0], shared_parm[0]);

          // forked children should exit through _exit()
          _exit(0);
        }
      }

      while(wait(nullptr) > 0)
      {
        // spin wait until all forked children complete
      }

      // return the result object via always_ready_future
      return agency::make_always_ready_future(std::move(result[0]));
    }
};


template<class ParallelPolicy>
int parallel_sum(ParallelPolicy&& policy, int* data, int n)
{
  // create a view of the input
  agency::experimental::span<int> input(data, n);

  // divide the input into 4 tiles
  int num_agents = 4;
  auto tiles = agency::experimental::tile_evenly(input, num_agents);

  // create agents to sum each tile in parallel
  auto partial_sums = agency::bulk_invoke(policy(num_agents), [=](agency::parallel_agent& self)
  {
    // get this parallel agent's tile
    auto this_tile = tiles[self.index()];

    // return the sum of this tile
    return std::accumulate(this_tile.begin(), this_tile.end(), 0);
  });

  // return the sum of partial sums
  return std::accumulate(partial_sums.begin(), partial_sums.end(), 0);
}


int main()
{
  std::vector<int> vec(32 << 20, 1);

  // compute a reference
  int reference  = parallel_sum(agency::par, vec.data(), vec.size());

  // now execute parallel_sum on our executor
  fork_executor fork_exec;

  int fork_sum = parallel_sum(agency::par.on(fork_exec), vec.data(), vec.size());

  // validate that the results match
  assert(reference == fork_sum);
  
  std::cout << "OK" << std::endl;
}

