from setuptools import setup, find_packages

setup(
    name='ml-project',
    version='1.0.0',
    description='Final Project',
    url='https://github.com/cis700/ml-project',
    author='Fanion Newsome',
    author_email='fnewsome@syr.edu',
    license='MIT',
    packages=find_packages(exclude=['src.test']),
    install_requires=['scikit-learn', 'numpy']
)
