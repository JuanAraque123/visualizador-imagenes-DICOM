"""
Utility module for the DICOM Viewer application.
"""
from .dicom_handler import load_dicom_file, DICOMLoadResult

__all__ = ['load_dicom_file', 'DICOMLoadResult'] 