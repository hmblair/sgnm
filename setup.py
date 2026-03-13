from setuptools import setup, find_packages

NAME = 'sgnm'
VERSION = '2.0.0'
LICENSE = 'CC BY-NC 4.0'
AUTHOR = 'Hamish M. Blair'
EMAIL = 'hmblair@stanford.edu'
URL = 'https://github.com/hmblair/sgnm'

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    license=LICENSE,
    install_requires=[
        'torch>=2.0',
        'ciffy',
    ],
    extras_require={
        'equivariant': ['flash-eq>=0.1.0'],
    },
)
