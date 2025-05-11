from setuptools import setup, find_packages

setup(
    name='easyocr',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch',
        'numpy',
        'opencv-python',
        'Pillow',
    ],
    entry_points={},
)
