# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 18:55:22 2021

@author: karan
"""

from setuptools import find_packages, setup
setup(
    name='prepdata',
    packages=find_packages(include=['prepdata']),
    version='0.1.0',
    description='Automating the process of Data Preprocessing for Data Science',
    author='Karan Malik',
    license='MIT',
    install_requires=['joblib==1.0.1','numpy==1.20.1','pandas==1.2.2','PyQt5==5.9.2','python-dateutil==2.8.1','pytz==2021.1','scikit-learn==0.24.1',
                      'scipy==1.6.1','setuptools==52.0.0.post20210125','sip==4.19.8','six==1.15.0','sklearn==0.0','threadpoolctl==2.1.0','wheel==0.36.2',
                      'wincertstore==0.2'],
    setup_requires=[],
    test_requires=[],
)