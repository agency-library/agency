#include <agency/agency.hpp>
#include <agency/cuda.hpp>

#include <vector>
#include <math.h>
#include <random>
#include <chrono>
#include <iostream>


__host__ __device__
float cnd(float x)
{
  static constexpr float sqrt2 = 1.41421356237; 
  static constexpr float inv_sqrt2 = 1.f / sqrt2;

  return 0.5f * (1.f + erf(inv_sqrt2 * x));
}


// XXX should really return something like pair<float,float>
__host__ __device__
void black_scholes(float stock_price,
                   float option_strike,
                   float option_years,
                   float riskless_rate,
                   float volatility,
                   float& call,
                   float& put)
{
  float sqrt_option_years = sqrtf(option_years);
  float d1 = (logf(stock_price / option_strike) + (riskless_rate + 0.5f * volatility * volatility) * option_years) / (volatility * sqrt_option_years);
  float d2 = d1 - volatility * sqrt_option_years;

  float exp_rt = exp(-riskless_rate * option_years);

  call = stock_price * cnd(d1) - option_strike * exp_rt * cnd(d2);
  put  = option_strike * exp_rt * (1.f - cnd(d2)) - stock_price * (1.f - cnd(d1));
}


auto grid(int num_blocks, int block_size) ->
  decltype(agency::cuda::par(num_blocks, agency::cuda::con(block_size)))
{
  return agency::cuda::par(num_blocks, agency::cuda::con(block_size));
}


using cuda_thread = agency::parallel_group<agency::concurrent_agent>;


void black_scholes(float* call_result, float* put_result,
                   const float* stock_price, const float* option_strike,
                   const float* option_years, 
                   float risk_free, float volatility, int n)
{
  using namespace agency;

  bulk_invoke(cuda::par(n), [=] __device__ (parallel_agent& self)
  {
    int i = self.index();

    black_scholes(stock_price[i],
                  option_strike[i],
                  option_years[i],
                  risk_free,
                  volatility,
                  call_result[i],
                  put_result[i]);
  });
}


void initialize_problem(int seed, int n, float *price, float *strike, float *years)
{
  std::default_random_engine rng(seed);
  std::uniform_real_distribution<float> random_price(5.f, 30.f);
  std::uniform_real_distribution<float> random_strike(1.f, 100.f);
  std::uniform_real_distribution<float> random_years(0.25f, 10.f);

  for(int i = 0; i < n; ++i)
  {
    price[i] = random_price(rng);
    strike[i] = random_strike(rng);
    years[i] = random_years(rng);
  }
}


int main()
{
  int n = 4000000;
  
  using vector = std::vector<float, agency::cuda::allocator<float>>;

  vector stock_price(n);
  vector option_strike(n);
  vector option_years(n);
  vector call_result(n);
  vector put_result(n);
  
  initialize_problem(13, n, stock_price.data(), option_strike.data(), option_years.data());

  float risk_free = 0.02f;
  float volatility = 0.30f;

  // warm up
  black_scholes(call_result.data(), put_result.data(), stock_price.data(), 
                option_strike.data(), option_years.data(),
                risk_free, volatility, n);
  
  auto start = std::chrono::high_resolution_clock::now();
  
  int num_trials = 100;
  
  for(int i = 0; i < num_trials; ++i)
  {
    black_scholes(call_result.data(), put_result.data(), stock_price.data(), 
                  option_strike.data(), option_years.data(),
                  risk_free, volatility, n);
  }
  
  std::chrono::nanoseconds elapsed = std::chrono::high_resolution_clock::now() - start;
  
  // convert total nsecs to mean seconds
  double seconds = (double(elapsed.count()) / num_trials) / 1000000000;

  double gigaoptions = double(2 * n) / (1 << 30);
  double gigabytes = double(5 * n * sizeof(float)) / (1 << 30);
  double bandwidth = gigabytes / seconds;
  double performance = gigaoptions / seconds;
  double ms = 1000.f * seconds;
  
  std::cout << "Mean time " << ms << " msec" << std::endl;
  std::cout << "Black-Scholes Bandwidth: " << bandwidth << " GB/s" << std::endl;
  std::cout << "Performance: " << performance << " GOptions/s" << std::endl;

  std::cout << "OK" << std::endl;

  return 0;
}

