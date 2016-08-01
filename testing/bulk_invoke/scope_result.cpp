#include <agency/agency.hpp>
#include <cassert>
#include <vector>

template<class ExecutionPolicy1, class ExecutionPolicy2>
void test(ExecutionPolicy1 outer, ExecutionPolicy2 inner)
{
  {
    // bulk_invoke with no parameters

    auto policy = outer(2, inner(3));
    using agent = typename decltype(policy)::execution_agent_type;

    auto result = agency::bulk_invoke(policy,
      [](agent& self) -> agency::scope_result<1,int>
    {
      if(self.inner().index() == 0)
      {
        return 7;
      }

      return std::ignore;
    });

    assert(result == std::vector<int>(2, 7));
  }

  {
    // bulk_invoke with one parameter

    auto policy = outer(2, inner(3));
    using agent = typename decltype(policy)::execution_agent_type;

    int val = 13;

    auto result = agency::bulk_invoke(policy,
      [](agent& self, int val) -> agency::scope_result<1,int>
    {
      if(self.inner().index() == 0)
      {
        return val;
      }

      return std::ignore;
    },
    val);

    assert(result == std::vector<int>(2, 13));
  }

  {
    // bulk_invoke with one shared parameter
    
    auto policy = outer(2, inner(3));
    using agent = typename decltype(policy)::execution_agent_type;

    int val = 13;

    auto result = agency::bulk_invoke(policy,
      [](agent& self, int& val) -> agency::scope_result<1,int>
    {
      if(self.inner().index() == 0)
      {
        return val;
      }

      return std::ignore;
    },
    agency::share(val));

    assert(result == std::vector<int>(2, 13));
  }
}

int main()
{
  using namespace agency;

  test(seq, seq);
  test(seq, par);
  test(con, seq);

  std::cout << "OK" << std::endl;

  return 0;
}

