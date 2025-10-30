# Visor DICOM

Un visor de imágenes DICOM con capacidades de procesamiento de imágenes implementado en Python usando PyQt6.

## Características

- Visualización de archivos DICOM individuales y múltiples
- Navegación entre cortes para imágenes multi-slice
- Ajuste de ventana/nivel
- Procesamiento de imágenes básico
- Interfaz gráfica intuitiva
- Soporte para metadata DICOM
- Herramientas de selección y medición

## Requisitos

- Python 3.8 o superior
- PyQt6
- pydicom
- numpy
- opencv-python

## Instalación

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd dicom-viewer
```

2. Crear un entorno virtual (recomendado):
```bash
python -m venv venv
```

3. Activar el entorno virtual:
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

4. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. Ejecutar la aplicación:
```bash
python main.py
```

2. Usar la interfaz gráfica para:
- Cargar archivos DICOM individuales
- Cargar directorios de archivos DICOM
- Navegar entre cortes
- Ajustar visualización
- Realizar mediciones y selecciones

## Estructura del Proyecto

```
dicom-viewer/
├── dicom_viewer/         # Código fuente principal
│   ├── __init__.py
│   ├── python_app.py    # Aplicación principal
│   └── resources/       # Recursos (imágenes, iconos, etc.)
├── requirements.txt     # Dependencias del proyecto
├── README.md           # Este archivo
└── main.py            # Punto de entrada
```

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT. 