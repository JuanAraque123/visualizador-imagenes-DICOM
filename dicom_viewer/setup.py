from setuptools import setup, find_packages

setup(
    name="dicom_viewer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'PyQt6',
        'pydicom',
        'opencv-python',
        'numpy',
    ],
)