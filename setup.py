from setuptools import setup, find_packages

setup(
    name='topometricloc',
    version='0.1.0',
    install_requires=[
        'numpy>=1.18.1',
        'scipy>=1.4.1',
        'pandas',
        'tqdm',
        'matplotlib',
        'ipykernel',
        'PyYAML',
        'scikit-learn'
    ],
    package_dir = {'': 'src/'}
)

