from setuptools import setup, find_packages

long_description = """coulomb_kmc is a FMM-KMC implementation that is built on PPMD."""

install_requires = []
with open('requirements.txt') as fh:
    for l in fh:
        if len(l) > 0:
            install_requires.append(l)



setup(
   name='coulomb_kmc',
   version='1.0',
   description='FMM-KMC implementation',
   license="GPL3",
   long_description=long_description,
   author='William R Saunders',
   author_email='W.R.Saunders@bath.ac.uk',
   url="https://bitbucket.org/wrs20/coulomb_kmc",
   packages=find_packages(),
   install_requires=install_requires,
   scripts=[],
   include_package_data=True
)
