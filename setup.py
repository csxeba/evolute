from setuptools import setup, find_packages

setup(
    name='evolute',
    version='0.9.0',
    packages=find_packages(),
    url='https://github.com/csxeba/evolute.git',
    license='MIT',
    author='Csaba Gór',
    author_email='csxeba@gmail.com',
    description='Evolutionary algorithm toolbox',
    long_description=open("Readme.md").read(),
    long_description_content_type='text/markdown'
)
