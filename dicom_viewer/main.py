#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import traceback

print("Iniciando la aplicación...")

# Asegurarse de que el directorio actual esté en el PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

print("Importando módulos...")
from PyQt5.QtWidgets import QApplication
from dicom_viewer.ui.dicom_viewer import DICOMViewer

def main():
    try:
        print("Creando la aplicación Qt...")
        # Crear la aplicación primero
        app = QApplication(sys.argv)
        
        # Establecer el estilo de la aplicación (opcional)
        app.setStyle('Fusion')
        
        print("Creando el visor DICOM...")
        # Crear y mostrar el visor DICOM después de crear QApplication
        viewer = DICOMViewer()
        print("Mostrando el visor...")
        viewer.show()
        
        print("Iniciando el loop principal...")
        # Ejecutar el loop principal de la aplicación
        return app.exec_()
    except Exception as e:
        print(f"Error en la aplicación: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 