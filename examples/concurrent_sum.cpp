#include <agency/agency.hpp>
#include <vector>

int sum(const std::vector<int>& data)
{
  using namespace agency;

  return bulk_invoke(con(data.size()), [&](concurrent_agent& self) -> single_result<int>
  {
    // copy data into scratch buffer
    shared_vector<int> scratch(self, data);

    auto i = self.index();
    auto n = scratch.size();

    while(n > 1)
    {
      if(i < n/2)
      {
        scratch[i] += scratch[n - i - 1];
      }

      // wait for every agent in the group to reach this point
      self.wait();

      // cut the number of active agents in half
      n -= n/2;
    }

    // the first agent returns the result
    if(i == 0)
    {
      return scratch[0];
    }

    // all other agents return an ignored value 
    return std::ignore;
  });
}

int main()
{
  int n = 10;

  std::vector<int> data(n, 1);

  auto result = sum(data);

  std::cout << "sum is " << result << std::endl;

  assert(result == n);

  std::cout << "OK" << std::endl;

  return 0;
}

