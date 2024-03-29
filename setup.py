from setuptools import setup

setup(
    name='iET',
    version='1.2.9',    
    description='iET - iterative Expected Target',
    url='https://github.com/dandolodavid/iET',
    author='Dandolo David',
    author_email='dandolodavid@gmail.com',
    license='BSD 4-clause',
    packages=['iET'],
    install_requires=['numpy', 'pandas', 'plotly > 4.0.0' ],

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Legal Industry',
        'Intended Audience :: Manufacturing',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
