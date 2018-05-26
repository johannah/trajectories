from setuptools import find_packages, setup, Command
import os
import io
import sys


# Package meta-data.
NAME = 'gym_trajectories'
DESCRIPTION = 'Model-based Planning'
URL = 'https://github.com/johannah/trajectories'
EMAIL = 'jh1736@gmail.com'
AUTHOR = 'Johanna Hansen'
REQUIRES_PYTHON = '>=2.7.0'
VERSION = None

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
        with open(os.path.join(here, NAME, '__version__.py')) as f:
                    exec(f.read(), about)
else:
        about['__version__'] = VERSION


setup(name='gym_trajectories',
      packages=find_packages(exclude=('tests',)),
      version=about['__version__'],
      author=AUTHOR,
      author_email=EMAIL,
      python_requires=REQUIRES_PYTHON,
      url=URL,
      )
