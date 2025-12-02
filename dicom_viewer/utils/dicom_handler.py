import pydicom
from pathlib import Path
from typing import Optional, Dict, Any

class DICOMHandler:
    def __init__(self):
        self.current_file: Optional[pydicom.FileDataset] = None
        self.metadata: Dict[str, Any] = {}
    
    def load_file(self, file_path: Path) -> bool:
        try:
            self.current_file = pydicom.dcmread(str(file_path))
            self._extract_metadata()
            return True
        except Exception as e:
            print(f"Error al cargar archivo DICOM: {e}")
            return False
    
    def _extract_metadata(self):
        if self.current_file:
            self.metadata = {
                'PatientName': str(self.current_file.get('PatientName', 'N/A')),
                'StudyDate': str(self.current_file.get('StudyDate', 'N/A')),
                'Modality': str(self.current_file.get('Modality', 'N/A')),
                'ImageSize': self.current_file.pixel_array.shape if hasattr(self.current_file, 'pixel_array') else None
            } 