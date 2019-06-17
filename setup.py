from setuptools import setup

setup(
    name='encoding_model',
    version='0.1',
    requires=['numpy', 'scipy', 'sklearn', 'statsmodels'],
    tests_require=['pytest', 'pytest-runner'],
    packages=['encoding_model'],
)
