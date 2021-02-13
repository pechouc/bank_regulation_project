from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='bank_regulation_project',
      version="1.0",
      description="Project Description",
      packages=find_packages(),
      test_suite='tests',
      # Include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
