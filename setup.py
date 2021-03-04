# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 18:55:22 2021

@author: karan
"""
import markdown

f = open('README.md', 'r')
htmlmarkdown=markdown.markdown( f.read() )

from setuptools import find_packages, setup
setup(
    name='PrepData',
    packages=find_packages(),
    version='0.1.12',
    description='Automating the process of Data Preprocessing for Data Science',
    author='Karan Malik',
    author_email='karanmalik2000@gmail.com',
    url="https://github.com/Karan-Malik/prepdata",
    license='MIT',
    long_description = htmlmarkdown,
    long_description_content_type = 'text/markdown',
    install_requires=['joblib==1.0.1','numpy==1.20.1','pandas==1.2.2','python-dateutil==2.8.1','pytz==2021.1','scikit-learn==0.24.1',
                      'scipy==1.6.1','setuptools==52.0.0','sip==6.0.1','six==1.15.0','sklearn==0.0','threadpoolctl==2.1.0','wheel==0.36.2',
                      'wincertstore==0.2','nltk==3.5'],
    setup_requires=[],
    test_requires=[],
)
