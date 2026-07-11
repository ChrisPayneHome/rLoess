from setuptools import setup, find_packages

setup(
    name='pyloess',
    version='1.0.0',
    url='https://github.com/ChrisPayneHome/Loess-model.git',
    author='Christian Payne',
    author_email='',
    description='Implementation of Loess model',
    packages=find_packages(),    
    install_requires=['numpy >= 1.11.1', 'flake8==7.3.0'],
)
