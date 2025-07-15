from setuptools import setup, find_packages

setup(
    name='AutomaticRange',
    version='0.1.1',
    author='Pacôme Prompsy',
    author_email='pacome.prompsy@unil.ch',
    description='A python package for automatic range prediction.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'napari[all]',
        'numpy',
        'torch',
        'scipy',
        'scikit-image'
    ],
)