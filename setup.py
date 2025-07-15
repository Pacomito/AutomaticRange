from setuptools import setup, find_packages

setup(
    name='napari-automatic-range',
    version='0.1.0',
    author='Pacôme Prompsy',
    author_email='pacome.prompsy@unil.ch',
    description='A Napari plugin for automatic range prediction.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'napari[all]',
        'numpy',
        'torch',
        'scipy',
        'scikit-image'
    ],
    entry_points={
    'napari.plugin': [
        'napari-automatic-range = napari_automatic_range',
    ],
},
)