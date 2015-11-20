import os

# command line variables
vars = Variables()
vars.Add('CXX', 'compiler', 'clang')

env = Environment(variables = vars)
env.MergeFlags(['-I.', '-std=c++11', '-lstdc++', '-lpthread'])

# find all sources in the current directory
sources = []
directories = ['.']
extensions = ['.cpp']

for dir in directories:
  for ext in extensions:
    regex = os.path.join(dir, '*' + ext)
    sources.extend(env.Glob(regex))

# create a test program for each source
tests = []
for src in sources:
  test = env.Program(src)
  # add the test to the 'run_tests' alias
  test_alias = env.Alias('run_tests', [test], test[0].abspath)
  # always build the 'run_tests' target whether or not it needs it
  env.AlwaysBuild(test_alias)

