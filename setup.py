from setuptools import setup, find_packages

setup(
    name='topometricloc',
    version='0.1.0',
    install_requires=[
        'numpy>=1.19',
        'scipy>=1.6.0',
        'opencv-python',
        'openvino',
        'pandas',
        'tqdm',
        'matplotlib',
        'ipykernel',
        'PyYAML',
        'scikit-learn',
        'openpyxl'
    ],
    package_dir = {'': 'src/'}
)

