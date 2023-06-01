from pathlib import Path
from setuptools import find_packages, setup

with open('README.md', encoding='utf8') as file:
    long_description = file.read()

setup(
    name='ste_gan',
    description='STE-GAN: Speech-to-Electromyography Signal Conversion using Generative Adversarial Networks',
    version='0.0.4',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['emg', 'speech', 'gan', 'pytorch']
)
