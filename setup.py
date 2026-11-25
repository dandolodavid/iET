from setuptools import setup

setup(
    name='iET',
    version='1.3.0',    
    description='iET - iterative Expected Target',
    url='https://github.com/dandolodavid/iET',
    author='Dandolo David',
    author_email='dandolodavid@gmail.com',
    license='BSD 4-clause',
    packages=['iET'],
    python_requires='>=3.8',
    install_requires=['numpy>=1.17', 'pandas>=1.1', 'plotly>=5.5'],

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Legal Industry',
        'Intended Audience :: Manufacturing',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
    ],
)
