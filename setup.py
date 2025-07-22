from setuptools import setup, find_packages

setup(
    name='AutomaticRange',
    version='0.1.6',
    author='Pacôme Prompsy',
    author_email='pacome.prompsy@unil.ch',
    description='A python package for automatic range prediction.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'scipy',
        'scikit-image',
        'tifffile',
        'opencv-python'
    ],
)