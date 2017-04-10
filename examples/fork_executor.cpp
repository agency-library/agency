#include <agency/agency.hpp>
#include <agency/experimental/span.hpp>
#include <agency/experimental/ranges/tile.hpp>

#include <algorithm>
#include <vector>

#include <unistd.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/wait.h>

template <typename T>
class mmap_allocator
{
public:
    using value_type = T;

    mmap_allocator() = default;

    template <class U>
    mmap_allocator(const mmap_allocator<U>&) {}

    T* allocate(std::size_t n)
    {
        if (n <= std::numeric_limits<std::size_t>::max() / sizeof(T))
        {
            return static_cast<T*>(mmap(NULL, n * sizeof (T),
                                        PROT_READ | PROT_WRITE,
                                        MAP_SHARED | MAP_ANONYMOUS,
                                        -1, 0));
        }
        throw std::bad_alloc();
    }
    void deallocate(T* ptr, std::size_t n)
    {
        munmap(ptr, sizeof(*ptr));
    }
};

template <typename T, typename U>
inline bool operator == (const mmap_allocator<T>&, const mmap_allocator<U>&)
{
    return true;
}

template <typename T, typename U>
inline bool operator != (const mmap_allocator<T>& a, const mmap_allocator<U>& b)
{
    return !(a == b);
}

class fork_executor
{
public:
    using execution_category = agency::parallel_execution_tag;
    template<typename T> using allocator = mmap_allocator<T>;

    template<class Function, class ResultFactory, class SharedFactory>
    agency::detail::result_of_t<ResultFactory()>
    bulk_sync_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory)
    {
        auto result = result_factory();
        auto shared_parm = shared_factory();

        for(size_t i = 0; i < n; ++i)
        {
            if(fork() == 0)
            {
                f(i, result, shared_parm);
                _exit(EXIT_SUCCESS);
            }
        }

        while (wait(NULL) > 0)
        {
            /* do nothing */
        }

        return std::move(result);
    }
};

template<class Executor>
int parallel_sum(Executor&& exec, int* data, int n)
{
    // create a view of the input
    agency::experimental::span<int> input(data, n);
    // divide the input into 4 tiles
    int num_agents = 4;
    auto tiles = agency::experimental::tile_evenly(input, num_agents);
    // create 8 agents to sum each tile in parallel
    auto partial_sums = agency::bulk_invoke(agency::par(num_agents).on(exec), [=](agency::parallel_agent& self)
    {
        // get this parallel agent's tile
        auto this_tile = tiles[self.index()];
        // return the sum of this tile
        return std::accumulate(this_tile.begin(), this_tile.end(), 0);
    });
    // return the sum of partial sums
    return std::accumulate(partial_sums.begin(), partial_sums.end(), 0);
}

int main(void)
{
    std::vector<int> vec(32 << 20, 1);
    int par_sum  = parallel_sum(agency::parallel_executor(), vec.data(), vec.size());
    int fork_sum = parallel_sum(fork_executor(), vec.data(), vec.size());
    assert(par_sum == fork_sum);

    return EXIT_SUCCESS;
}
