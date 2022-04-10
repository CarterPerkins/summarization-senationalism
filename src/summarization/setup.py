from setuptools import setup, find_packages

setup(name='summarization_local_module',
      version='0.1',
      packages=find_packages(),
     )

# if you cannot install the summarization package
# on your system, just add the path to this directory
# to the PYTHONPATH
# export PYTHONPATH=/path/to/summarization:$PYTHONPATH
