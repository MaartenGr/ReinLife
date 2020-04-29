# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ReinLife',
    version='1.0.0',
    description='Creating Artificial Life with Reinforcement Learning',
    long_description=readme,
    author='Maarten Grootendorst',
    author_email='maartengrootendorst@gmail.com',
    url='https://github.com/MaartenGr/TheGame',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
