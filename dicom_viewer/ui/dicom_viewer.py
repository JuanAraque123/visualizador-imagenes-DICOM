import os
import cv2
import numpy as np
import pydicom
import traceback
import subprocess
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'processing'))
from HeartSegmentation import predict
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QComboBox, QStatusBar, QMessageBox, QProgressDialog,
                           QSizePolicy, QSlider, QMenuBar, QDialog, QScrollArea,
                           QListWidget, QListWidgetItem, QAction, QFrame)
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent
from datetime import datetime
import sys
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

def is_valid_dicom(file_path):
    """Verificar si un archivo es DICOM válido"""

    
    try:
        # Verificar si el archivo es lo suficientemente grande para ser DICOM
        if os.path.getsize(file_path) < 132:  # Tamaño mínimo para un header DICOM
            return False
            
        # Intentar leer el archivo DICOM directamente
        try:
            dicom = pydicom.dcmread(file_path, force=True)
            if hasattr(dicom, 'pixel_array') and dicom.pixel_array.size > 0:
                return True
        except:
            pass
        
        # Si falla, verificar el header
        with open(file_path, 'rb') as f:
            header = f.read(132)
            # Verificar si contiene la firma DICOM
            if b'DICM' in header or b'DICOM' in header:
                try:
                    dicom = pydicom.dcmread(file_path, force=True)
                    return hasattr(dicom, 'pixel_array') and dicom.pixel_array.size > 0
                except:
                    pass
        
        # Verificar si es un archivo de recorte
        if os.path.basename(file_path).startswith('selection_'):
            try:
                dicom = pydicom.dcmread(file_path, force=True)
                return hasattr(dicom, 'pixel_array') and dicom.pixel_array.size > 0
            except:
                pass
        
        return False
    except Exception as e:
        print(f"Error al verificar archivo {file_path}: {str(e)}")
        return False

def process_batch(batch_files, directory):
    """Procesar un lote de archivos"""
    valid_files = []
    for file_name in batch_files:
        file_path = os.path.join(directory, file_name)
        if is_valid_dicom(file_path):
            valid_files.append(file_path)
    return valid_files

# Remove torch dependencies since they're causing numpy compatibility issues
# We'll implement a simpler analysis approach

class SelectionTool:
    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.is_drawing = False
        self.selection_type = "rectangular"  # o "circular"
        self.current_overlay = None  # Para almacenar la imagen con la selección dibujada
        self.final_selection = None  # Para almacenar la selección final
        self.mask_buffer = None  # Buffer para máscaras circulares
    
    def start_selection(self, point):
        self.start_point = (point.x(), point.y())
        self.is_drawing = True
        self.final_selection = None  # Limpiar selección anterior
    
    def update_selection(self, point):
        self.end_point = (point.x(), point.y())
    
    def end_selection(self):
        self.is_drawing = False
        # Guardar la selección final
        if self.start_point and self.end_point:
            if self.selection_type == "rectangular":
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                x = min(x1, x2)
                y = min(y1, y2)
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                self.final_selection = ("rectangular", (x, y, width, height))
            else:  # circular
                center = self.start_point
                radius = int(((self.end_point[0] - self.start_point[0])**2 + 
                            (self.end_point[1] - self.start_point[1])**2)**0.5)
                self.final_selection = ("circular", (center, radius))
    
    def draw_selection(self, image):
        """Dibujar la selección actual sobre la imagen usando OpenCV"""
        # Reutilizar buffer si es posible
        if self.current_overlay is None or self.current_overlay.shape != image.shape:
            self.current_overlay = image.copy()
        else:
            self.current_overlay[:] = image
        
        # Dibujar la selección actual si se está dibujando
        if self.is_drawing and self.start_point and self.end_point:
            if self.selection_type == "rectangular":
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                x = min(x1, x2)
                y = min(y1, y2)
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                cv2.rectangle(self.current_overlay, (x, y), (x + width, y + height), (255, 0, 0), 1)
            else:  # circular
                center = self.start_point
                radius = int(((self.end_point[0] - self.start_point[0])**2 + 
                            (self.end_point[1] - self.start_point[1])**2)**0.5)
                cv2.circle(self.current_overlay, center, radius, (255, 0, 0), 1)
        
        # Dibujar la selección final si existe
        if self.final_selection:
            selection_type, selection_data = self.final_selection
            if selection_type == "rectangular":
                x, y, width, height = selection_data
                cv2.rectangle(self.current_overlay, (x, y), (x + width, y + height), (255, 0, 0), 1)
            else:  # circular
                center, radius = selection_data
                cv2.circle(self.current_overlay, center, radius, (255, 0, 0), 1)
        
        return self.current_overlay
    
    def get_selection(self):
        """Obtener la selección actual o final"""
        if self.final_selection:
            return self.final_selection[1]
        return None
        
    def extract_selection(self, image):
        """Extraer la región seleccionada de la imagen"""
        if not self.final_selection:
            return None
            
        selection_type, selection_data = self.final_selection
        
        # Asegurarse de que la imagen esté en el formato correcto
        if len(image.shape) == 2:  # Imagen en escala de grises
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if selection_type == "rectangular":
            x, y, width, height = selection_data
            # Asegurarse de que los valores estén dentro de los límites de la imagen
            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            width = min(width, image.shape[1] - x)
            height = min(height, image.shape[0] - y)
            
            if width <= 0 or height <= 0:
                return None
                
            return image[y:y+height, x:x+width]
            
        else:  # circular
            center, radius = selection_data
            # Crear máscara circular de manera eficiente
            if self.mask_buffer is None or self.mask_buffer.shape != image.shape[:2]:
                self.mask_buffer = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # Limpiar buffer anterior
            self.mask_buffer.fill(0)
            
            # Dibujar círculo en la máscara
            cv2.circle(self.mask_buffer, center, radius, 1, -1)
            
            # Extraer la región circular de manera eficiente
            subimage = image.copy()
            subimage[self.mask_buffer == 0] = 0
            
            # Encontrar los límites del círculo de manera vectorizada
            rows = np.any(self.mask_buffer, axis=1)
            cols = np.any(self.mask_buffer, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                return None
                
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            return subimage[rmin:rmax+1, cmin:cmax+1]

class SubimageViewer(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Visualización de Subimágenes")
        self.setMinimumSize(800, 600)
        
        # Variables para el zoom
        self.zoom_factor = 1.0
        self.zoom_step = 0.1
        
        # Layout principal
        layout = QVBoxLayout()
        
        # Área de visualización con scroll
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)
        
        # Widget contenedor para las subimágenes
        self.subimages_container = QWidget()
        self.subimages_layout = QVBoxLayout(self.subimages_container)
        self.scroll_area.setWidget(self.subimages_container)
        
        # Controles de zoom
        zoom_controls = QHBoxLayout()
        self.zoom_in_button = QPushButton("Zoom +")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button = QPushButton("Zoom -")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.reset_zoom_button = QPushButton("Reset Zoom")
        self.reset_zoom_button.clicked.connect(self.reset_zoom)
        
        zoom_controls.addWidget(self.zoom_in_button)
        zoom_controls.addWidget(self.zoom_out_button)
        zoom_controls.addWidget(self.reset_zoom_button)
        layout.addLayout(zoom_controls)
        
        self.setLayout(layout)
    
    def display_subimages(self, subimages):
        """Mostrar todas las subimágenes seleccionadas"""
        # Limpiar layout actual
        for i in reversed(range(self.subimages_layout.count())): 
            self.subimages_layout.itemAt(i).widget().setParent(None)
        
        # Agregar cada subimagen con su etiqueta
        for frame_key, subimage in subimages.items():
            frame_num = frame_key.split('_')[1]
            
            # Crear contenedor para la subimagen
            container = QWidget()
            container_layout = QVBoxLayout(container)
            
            # Etiqueta con el número de frame
            label = QLabel(f"Frame {frame_num}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(label)
            
            # Convertir subimagen a QPixmap
            height, width = subimage.shape
            bytes_per_line = width
            qimage = QImage(subimage.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            
            # Label para mostrar la imagen
            image_label = QLabel()
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(image_label)
            
            self.subimages_layout.addWidget(container)
    
    def zoom_in(self):
        """Aumentar el nivel de zoom"""
        if self.current_image is None:
            return
            
        new_zoom = min(self.zoom_factor + self.zoom_step, self.max_zoom)
        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            self.display_image()
            
            # Reposicionar el cuadro de información
            self.position_image_info_label()
            
            # Actualizar información de coordenadas si el mouse está sobre la imagen
            if hasattr(self, 'image_label') and self.image_label.underMouse():
                mouse_pos = self.image_label.mapFromGlobal(self.cursor().pos())
                self.update_coordinates_display(mouse_pos)
    
    def zoom_out(self):
        """Disminuir el nivel de zoom"""
        if self.current_image is None:
            return
            
        new_zoom = max(self.zoom_factor - self.zoom_step, self.min_zoom)
        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            self.display_image()
            
            # Reposicionar el cuadro de información
            self.position_image_info_label()
            
            # Actualizar información de coordenadas si el mouse está sobre la imagen
            if hasattr(self, 'image_label') and self.image_label.underMouse():
                mouse_pos = self.image_label.mapFromGlobal(self.cursor().pos())
                self.update_coordinates_display(mouse_pos)
    
    def reset_zoom(self):
        """Resetear el zoom a su valor inicial"""
        if self.current_image is None:
            return
            
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.display_image()
        
        # Reposicionar el cuadro de información
        self.position_image_info_label()
        
        # Actualizar información de coordenadas si el mouse está sobre la imagen
        if hasattr(self, 'image_label') and self.image_label.underMouse():
            mouse_pos = self.image_label.mapFromGlobal(self.cursor().pos())
            self.update_coordinates_display(mouse_pos)
    
    def apply_zoom(self):
        """Aplicar el factor de zoom actual a todas las imágenes"""
        for i in range(self.subimages_layout.count()):
            container = self.subimages_layout.itemAt(i).widget()
            image_label = container.findChild(QLabel, "", Qt.FindChildOption.FindChildrenRecursively)
            if image_label and image_label.pixmap():
                original_pixmap = image_label.pixmap()
                scaled_pixmap = original_pixmap.scaled(
                    original_pixmap.width() * self.zoom_factor,
                    original_pixmap.height() * self.zoom_factor,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                image_label.setPixmap(scaled_pixmap)

class DICOMViewer(QMainWindow):
    instances = []  # Lista para mantener registro de todas las instancias

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visor DICOM con Procesamiento de Imágenes")
        
        # Establecer tamaños mínimos y máximos razonables
        self.setMinimumSize(1024, 768)
        
        # Inicializar variables
        self.current_image = None
        self.original_image = None
        self.current_slice = 0
        self.total_slices = 0
        self.is_playing = False
        self.current_directory_files = []  # Lista para mantener los archivos del directorio actual
        self.current_file_index = 0  # Índice del archivo actual
        
        # Configurar ruta por defecto para guardar selecciones
        self.default_save_path = os.path.join(
            os.path.expanduser("~"),
            "Documents",
            "DICOM_Selections"
        )
        # Crear el directorio si no existe
        if not os.path.exists(self.default_save_path):
            os.makedirs(self.default_save_path)
        
        # Variables para el zoom
        self.zoom_factor = 1.0
        self.zoom_step = 0.1
        self.max_zoom = 5.0
        self.min_zoom = 0.1
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.last_pan_pos = None
        
        # Setup playback timer
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.next_slice)
        
        # Inicializar variables para TensorFlow
        self.tf_model = None
        self.tf_tensors = None
        
        self.segmentation_model = None
        
        # Inicializar herramientas de selección
        self.selection_tool = SelectionTool()
        self.current_selections = {}  # Diccionario para guardar selecciones por frame
        self.selected_subimages = {}  # Diccionario para guardar subimágenes seleccionadas
        self.selection_sequence = []  # Lista para mantener el orden de las selecciones
        self.current_dicom_metadata = None  # Para almacenar los metadatos DICOM actuales
        
        self.setup_menu()
        self.setup_ui()
        
        # Posicionar inicialmente el cuadro de información
        self.position_image_info_label()
        
        # Agregar esta instancia a la lista
        DICOMViewer.instances.append(self)
        
        # Buffers preasignados para optimización
        self.image_buffer = None
        self.overlay_buffer = None
        self.coord_scale = (1.0, 1.0)
        
        # Sistema de caché para imágenes procesadas
        self.image_cache = {}
        self.cache_size = 10  # Número máximo de imágenes en caché
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Timer para actualizaciones diferidas
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.deferred_update)
        self.pending_update = False
        
        # Inicializar en modo maximizado después de crear la ventana
        self.showMaximized()

    def setup_menu(self):
        """Configurar la barra de menú"""
        menubar = self.menuBar()
        
        # Menú Archivo
        file_menu = menubar.addMenu('Archivo')
        
        new_window = QAction('Nueva Ventana', self)
        new_window.setShortcut('Ctrl+N')
        new_window.triggered.connect(self.create_new_window)
        file_menu.addAction(new_window)
        
        open_file = QAction('Abrir DICOM', self)
        open_file.setShortcut('Ctrl+O')
        open_file.triggered.connect(self.load_dicom)
        file_menu.addAction(open_file)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Salir', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Menú Herramientas
        tools_menu = menubar.addMenu('Herramientas')
        
        reset_view = QAction('Restablecer Vista', self)
        reset_view.triggered.connect(self.reset_view)
        tools_menu.addAction(reset_view)
        
        # Menú Ayuda
        help_menu = menubar.addMenu('Ayuda')
        
        about_action = QAction('Acerca de', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_new_window(self):
        """Crear una nueva ventana del visor DICOM"""
        new_viewer = DICOMViewer()
        new_viewer.show()
        # La ventana se mantendrá en memoria debido a la lista instances
        
    def closeEvent(self, event):
        """Manejar el cierre de la ventana"""
        try:
            # Detener timers
            self.play_timer.stop()
            self.update_timer.stop()
            
            # Limpiar caché y buffers
            self.clear_cache()
            self.image_buffer = None
            self.overlay_buffer = None
            
            # Limpiar selecciones
            self.clear_selections()
            self.selected_subimages.clear()
            self.selection_sequence.clear()
            
            # Limpiar imágenes
            self.current_image = None
            self.original_image = None
            
            # Remover de la lista de instancias
            DICOMViewer.instances.remove(self)
            
        except Exception as e:
            print(f"Error al cerrar la ventana: {str(e)}")
        finally:
            event.accept()
            
    def __del__(self):
        """Destructor para limpieza final"""
        try:
            self.clear_cache()
            self.image_buffer = None
            self.overlay_buffer = None
        except:
            pass

    def reset_view(self):
        """Restablecer la vista a su estado original"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.current_slice = 0
            if len(self.current_image.shape) == 3:
                self.slice_slider.setValue(0)
            # Eliminando la referencia al filter_selector que no existe
            self.display_image()
            self.statusBar.showMessage("Vista restablecida")
            
            # Restablecer controles de reproducción
            self.is_playing = False
            self.play_button.setText("Reproducir")
            self.play_timer.stop()
            self.speed_selector.setCurrentText("1 fps")
            
            # Resetear zoom y pan
            self.zoom_factor = 1.0
            self.pan_offset = QPoint(0, 0)
            self.display_image()
            
    def show_about(self):
        """Mostrar información sobre la aplicación"""
        # Crear el diálogo About
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("Acerca de DICOM Viewer")
        about_dialog.setFixedSize(600, 400)
        
        # Crear layout vertical
        layout = QVBoxLayout()
        
        # Agregar logo
        logo_label = QLabel()
        # Obtener la ruta absoluta del directorio actual y construir la ruta al logo
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(current_dir, "resources", "logo_unbosque.jpg")
        try:
            logo_pixmap = QPixmap(logo_path)
            if not logo_pixmap.isNull():
                scaled_pixmap = logo_pixmap.scaled(200, 200, 
                                                 Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation)
                logo_label.setPixmap(scaled_pixmap)
            else:
                logo_label.setText("Logo no disponible")
        except Exception as e:
            logo_label.setText("Logo no disponible")
        
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo_label)
        
        # Agregar texto
        info_text = QLabel(
            """<div style='text-align: center;'>
            <h3>DICOM Viewer con Procesamiento de Imágenes</h3>
            <p><b>Proyecto de investigación:</b></p>
            <p>Modelo computacional para determinar el puntaje de calcio basado en datos 
            de la gammagrafía de perfusión miocárdica con SPECT/CT como método para 
            mejorar el diagnóstico de riesgo de enfermedad coronaria.</p>
            <p><b>Investigadores Principales:</b></p>
            <p>Dr. Augusto Llamas<br>
            Ing. Fran Ernesto Romero Álvarez</p>
            <p><b>Auxiliares de investigación:</b></p>
            <p>Juan Pablo Araque<br>
            Alejandro Fonnegra</p>
            <p><b>Características:</b></p>
            <ul>
                <li>Visualización de imágenes DICOM</li>
                <li>Reproducción de secuencias</li>
                <li>Filtros de procesamiento de imágenes</li>
                <li>Análisis básico de imágenes</li>
            </ul>
            <p>Versión 1.0</p>
            </div>"""
        )
        info_text.setWordWrap(True)
        info_text.setOpenExternalLinks(True)
        layout.addWidget(info_text)
        
        # Botón de cerrar
        close_button = QPushButton("Cerrar")
        close_button.clicked.connect(about_dialog.close)
        layout.addWidget(close_button)
        
        about_dialog.setLayout(layout)
        about_dialog.exec()

    def toggle_play(self):
        """Alternar reproducción de la secuencia de imágenes"""
        if self.current_image is None or not self.current_directory_files:
            return
            
        if not self.is_playing:
            self.is_playing = True
            self.play_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_MediaPause))
            self.play_button.setToolTip("Pausar")
            self.stop_button.setEnabled(True)
            
            # Iniciar reproducción según la velocidad seleccionada
            speed_text = self.speed_selector.currentText()
            if speed_text == "2 fps":
                self.play_timer.start(500)
            elif speed_text == "1 fps":
                self.play_timer.start(1000)
            elif speed_text == "0.5 fps":
                self.play_timer.start(2000)
        else:
            self.pause_playback()

    def setup_ui(self):
        """Configurar la interfaz de usuario"""
        # Crear layout principal
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)  # Espacio entre paneles
        main_layout.setContentsMargins(10, 10, 10, 10)  # Márgenes externos
        
        # Panel izquierdo para lista de archivos
        left_panel = QWidget()
        left_panel.setMinimumWidth(200)  # Ancho mínimo para el panel izquierdo
        left_panel.setMaximumWidth(300)  # Ancho máximo para el panel izquierdo
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Lista de archivos DICOM
        self.file_list = QListWidget()
        self.file_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.file_list.itemClicked.connect(self.load_dicom_file)  # Restaurar conexión
        left_layout.addWidget(self.file_list)
        
        # Botón para cargar directorio
        self.load_dir_button = QPushButton("Cargar Directorio")
        self.load_dir_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.load_dir_button.clicked.connect(self.load_dicom_directory)  # Restaurar conexión
        left_layout.addWidget(self.load_dir_button)
        
        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel)
        
        # Panel central para la visualización
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(5, 5, 5, 5)
        
        # Crear barra de herramientas con scroll horizontal
        toolbar_scroll = QScrollArea()
        toolbar_scroll.setWidgetResizable(True)
        toolbar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        toolbar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        toolbar_scroll.setFixedHeight(50)  # Altura fija para la barra de herramientas
        
        toolbar_widget = QWidget()
        self.toolbar = QHBoxLayout(toolbar_widget)
        self.toolbar.setSpacing(5)
        self.toolbar.setContentsMargins(5, 5, 5, 5)
        
        # Agregar botones de navegación con iconos
        self.prev_slice_button = QPushButton()
        self.prev_slice_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_MediaSeekBackward))
        self.prev_slice_button.setToolTip("Anterior")
        self.prev_slice_button.setFixedSize(32, 32)
        self.prev_slice_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.prev_slice_button.clicked.connect(self.previous_slice)
        self.prev_slice_button.setEnabled(False)
        self.toolbar.addWidget(self.prev_slice_button)
        
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_MediaPlay))
        self.play_button.setToolTip("Reproducir")
        self.play_button.setFixedSize(32, 32)
        self.play_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setEnabled(False)
        self.toolbar.addWidget(self.play_button)
        
        self.stop_button = QPushButton()
        self.stop_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_MediaStop))
        self.stop_button.setToolTip("Detener")
        self.stop_button.setFixedSize(32, 32)
        self.stop_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.stop_button.clicked.connect(self.stop_playback)
        self.stop_button.setEnabled(False)
        self.toolbar.addWidget(self.stop_button)
        
        self.next_slice_button = QPushButton()
        self.next_slice_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_MediaSeekForward))
        self.next_slice_button.setToolTip("Siguiente")
        self.next_slice_button.setFixedSize(32, 32)
        self.next_slice_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.next_slice_button.clicked.connect(self.next_slice)
        self.next_slice_button.setEnabled(False)
        self.toolbar.addWidget(self.next_slice_button)
        
        # Agregar selector de velocidad
        self.speed_selector = QComboBox()
        self.speed_selector.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.speed_selector.addItems(["2 fps", "1 fps", "0.5 fps"])  # Restaurar items
        self.speed_selector.setEnabled(False)
        self.speed_selector.currentTextChanged.connect(self.change_playback_speed)  # Restaurar conexión
        self.toolbar.addWidget(self.speed_selector)
        
        # Agregar slider para navegación
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.slice_slider.valueChanged.connect(self.slider_changed)  # Conectar el evento de cambio
        self.toolbar.addWidget(self.slice_slider)
        
        # Agregar etiqueta de slice
        self.slice_label = QLabel("Corte: 0/0")
        self.slice_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.toolbar.addWidget(self.slice_label)
        
        # Agregar controles de zoom
        self.zoom_in_button = QPushButton("Zoom +")
        self.zoom_in_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.zoom_in_button.clicked.connect(self.handle_zoom_in)  # Restaurar conexión
        self.toolbar.addWidget(self.zoom_in_button)
        
        self.zoom_out_button = QPushButton("Zoom -")
        self.zoom_out_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.zoom_out_button.clicked.connect(self.handle_zoom_out)  # Restaurar conexión
        self.toolbar.addWidget(self.zoom_out_button)
        
        self.reset_zoom_button = QPushButton("Reset Zoom")
        self.reset_zoom_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.reset_zoom_button.clicked.connect(self.handle_reset_zoom)  # Restaurar conexión
        self.toolbar.addWidget(self.reset_zoom_button)
        
        # Agregar separador visual
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.toolbar.addWidget(separator)
        
        # Agregar menú desplegable de filtros
        self.filters_label = QLabel("Filtros:")
        self.filters_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.toolbar.addWidget(self.filters_label)
        
        self.filters_combo = QComboBox()
        self.filters_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.filters_combo.addItems(["Sin filtro", "Blur", "Edge", "Sharpness", "Full Dynamic", "Skull"])
        self.filters_combo.currentTextChanged.connect(self.apply_image_filter)
        self.filters_combo.setEnabled(False)  # Solo habilitar cuando hay imagen
        self.toolbar.addWidget(self.filters_combo)
        
        # Botón para resetear filtros
        self.reset_filters_button = QPushButton("Reset Filtros")
        self.reset_filters_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.reset_filters_button.clicked.connect(self.reset_filters)
        self.reset_filters_button.setEnabled(False)  # Solo habilitar cuando hay imagen
        self.toolbar.addWidget(self.reset_filters_button)
        
        # Agregar separador visual
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.VLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        self.toolbar.addWidget(separator2)
        
        # Botón para ejecutar modelo predictivo
        self.predictive_model_button = QPushButton("Ejecutar Modelo Predictivo")
        self.predictive_model_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.predictive_model_button.clicked.connect(self.execute_predictive_model)
        self.predictive_model_button.setEnabled(False)  # Solo habilitar cuando hay imagen
        self.toolbar.addWidget(self.predictive_model_button)
        
        toolbar_widget.setLayout(self.toolbar)
        toolbar_scroll.setWidget(toolbar_widget)
        center_layout.addWidget(toolbar_scroll)
        
        # Contenedor para la imagen con scroll
        image_scroll = QScrollArea()
        image_scroll.setWidgetResizable(True)
        image_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        image_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        
        # Crear un widget contenedor para la imagen y las coordenadas
        image_widget = QWidget()
        image_widget.setLayout(QVBoxLayout())
        image_widget.layout().setContentsMargins(0, 0, 0, 0)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setMouseTracking(True)
        self.image_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.image_label.setStyleSheet("background-color: black;")
        image_widget.layout().addWidget(self.image_label)
        
        # Widget para mostrar información completa de la imagen (estilo MicroDicom)
        self.image_info_label = QLabel()
        self.image_info_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 200);
                color: white;
                padding: 8px 12px;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                font-weight: bold;
                line-height: 1.2;
            }
        """)
        self.image_info_label.setFixedSize(200, 180)
        self.image_info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.image_info_label.setWordWrap(True)
        
        # Texto inicial
        self.image_info_label.setText("Cargando imagen...")

        # Agregar el widget de información como hijo del image_widget para superposición
        self.image_info_label.setParent(image_widget)
        self.image_info_label.raise_()  # Asegurar que esté por encima de la imagen
        
        image_layout.addWidget(image_widget)
        
        image_container.setLayout(image_layout)
        image_scroll.setWidget(image_container)
        center_layout.addWidget(image_scroll, stretch=1)
        
        center_panel.setLayout(center_layout)
        main_layout.addWidget(center_panel, stretch=2)
        
        # Panel derecho para metadata
        right_panel = QWidget()
        right_panel.setMinimumWidth(250)  # Ancho mínimo para el panel derecho
        right_panel.setMaximumWidth(350)  # Ancho máximo para el panel derecho
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # Panel de metadata con scroll
        metadata_scroll = QScrollArea()
        metadata_scroll.setWidgetResizable(True)
        metadata_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        metadata_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        metadata_panel = QWidget()
        metadata_layout = QVBoxLayout(metadata_panel)
        metadata_layout.setContentsMargins(5, 5, 5, 5)
        
        # Título para metadata
        metadata_title = QLabel("Información del Paciente")
        metadata_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        metadata_layout.addWidget(metadata_title)
        
        # Área de texto para metadata
        self.metadata_text = QLabel()
        self.metadata_text.setWordWrap(True)
        self.metadata_text.setStyleSheet("background-color: white; padding: 10px; border: 1px solid #ccc;")
        self.metadata_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        metadata_layout.addWidget(self.metadata_text)
        
        metadata_panel.setLayout(metadata_layout)
        metadata_scroll.setWidget(metadata_panel)
        right_layout.addWidget(metadata_scroll)
        
        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel)
        
        # Agregar barra de estado
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # Configurar widget central
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Configurar herramientas de selección después de crear image_label
        self.setup_selection_tools()

    def handle_zoom_in(self):
        """Manejador para el botón de zoom in"""
        if self.current_image is None:
            return
            
        new_zoom = min(self.zoom_factor + self.zoom_step, self.max_zoom)
        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            # Si es el primer zoom, centrar la imagen
            if self.zoom_factor > 1.0 and self.pan_offset == QPoint(0, 0):
                self.pan_offset = QPoint(0, 0)
            self.display_image()

    def handle_zoom_out(self):
        """Manejador para el botón de zoom out"""
        if self.current_image is None:
            return
            
        new_zoom = max(self.zoom_factor - self.zoom_step, self.min_zoom)
        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            # Si volvemos a zoom 1.0, resetear el pan
            if self.zoom_factor <= 1.0:
                self.pan_offset = QPoint(0, 0)
            self.display_image()

    def handle_reset_zoom(self):
        """Manejador para el botón de reset zoom"""
        if self.current_image is None:
            return
            
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.display_image()

    def mouse_press_event(self, event):
        """Manejar el evento de presión del mouse"""
        if event.button() == Qt.MouseButton.MiddleButton or (event.button() == Qt.MouseButton.LeftButton and not self.is_selection_active()):
            self.is_panning = True
            self.last_pan_pos = event.pos()
        elif event.button() == Qt.MouseButton.LeftButton and self.is_selection_active():
            pos = self.calculate_image_coordinates(event.pos())
            if pos:
                self.selection_tool.start_selection(pos)
                self.update_image_with_selection()

    def mouse_move_event(self, event):
        """Manejar el evento de movimiento del mouse"""
        # Actualizar coordenadas en tiempo real
        self.update_coordinates_display(event.pos())
        
        if self.is_panning and self.last_pan_pos is not None:
            # Calcular el desplazamiento
            delta = event.pos() - self.last_pan_pos
            
            # Actualizar el offset de pan
            self.pan_offset += delta
            self.last_pan_pos = event.pos()
            
            # Calcular límites basados en el tamaño de la imagen y el zoom
            if self.current_image is not None:
                img_width = self.current_image.shape[1]
                img_height = self.current_image.shape[0]
                label_size = self.image_label.size()
                
                # Calcular el tamaño de la imagen escalada
                scaled_width = int(img_width * self.zoom_factor)
                scaled_height = int(img_height * self.zoom_factor)
                
                # Calcular los límites máximos de desplazamiento
                max_x = max(0, (scaled_width - label_size.width()) // 2)
                max_y = max(0, (scaled_height - label_size.height()) // 2)
                
                # Limitar el desplazamiento
                self.pan_offset.setX(max(-max_x, min(max_x, self.pan_offset.x())))
                self.pan_offset.setY(max(-max_y, min(max_y, self.pan_offset.y())))
            
            # Actualizar la imagen sin reiniciar el zoom
            self.update_image_with_selection()
        elif self.selection_tool.is_drawing and self.is_selection_active():
            pos = self.calculate_image_coordinates(event.pos())
            if pos:
                self.selection_tool.update_selection(pos)
                self.update_image_with_selection()

    def mouse_release_event(self, event):
        """Manejar el evento de liberación del mouse"""
        if event.button() == Qt.MouseButton.MiddleButton or (event.button() == Qt.MouseButton.LeftButton and not self.is_selection_active()):
            self.is_panning = False
            self.last_pan_pos = None
        elif event.button() == Qt.MouseButton.LeftButton and self.is_selection_active():
            pos = self.calculate_image_coordinates(event.pos())
            if pos:
                self.selection_tool.end_selection()
                self.update_image_with_selection()

    def update_image_with_selection(self):
        """Actualizar la imagen mostrada con la selección actual"""
        if self.current_image is None:
            return

        try:
            # Obtener la imagen actual
            if len(self.current_image.shape) == 3:  # Imagen 3D
                current_slice = self.current_image[self.current_slice]
            else:  # Imagen 2D
                current_slice = self.current_image
                
            # Reutilizar buffer si es posible
            if self.image_buffer is None or self.image_buffer.shape != current_slice.shape:
                self.image_buffer = current_slice.copy()
            else:
                self.image_buffer[:] = current_slice
                
            # Convertir a RGB si es necesario
            if len(self.image_buffer.shape) == 2:  # Imagen en escala de grises
                if self.overlay_buffer is None or self.overlay_buffer.shape != (*self.image_buffer.shape, 3):
                    self.overlay_buffer = cv2.cvtColor(self.image_buffer, cv2.COLOR_GRAY2RGB)
                else:
                    self.overlay_buffer[:] = cv2.cvtColor(self.image_buffer, cv2.COLOR_GRAY2RGB)
            else:
                self.overlay_buffer = self.image_buffer
                
            # Dibujar la selección
            overlay = self.selection_tool.draw_selection(self.overlay_buffer)
            
            # Convertir a QPixmap y mostrar
            height, width = overlay.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(overlay.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # Obtener dimensiones del label
            label_size = self.image_label.size()
            
            # Calcular el tamaño de la imagen con zoom
            zoomed_size = QSize(
                int(width * self.zoom_factor),
                int(height * self.zoom_factor)
            )
            
            # Escalar la imagen con zoom manteniendo la calidad
            scaled_pixmap = pixmap.scaled(
                zoomed_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Crear un nuevo pixmap del tamaño del label
            final_pixmap = QPixmap(label_size)
            final_pixmap.fill(Qt.GlobalColor.transparent)
            
            # Calcular la posición de la imagen con pan
            x = int((label_size.width() - scaled_pixmap.width()) // 2 + self.pan_offset.x())
            y = int((label_size.height() - scaled_pixmap.height()) // 2 + self.pan_offset.y())
            
            # Dibujar la imagen escalada en la posición correcta
            painter = QPainter(final_pixmap)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawPixmap(x, y, scaled_pixmap)
            
            # Si hay una selección activa, dibujarla
            if self.is_selection_active():
                # Configurar el renderizado para círculos suaves
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setRenderHint(QPainter.RenderHint.HighQualityAntialiasing)
                
                # Dibujar selección existente
                if self.selection_tool.final_selection:
                    selection_type, selection_data = self.selection_tool.final_selection
                    if selection_type == "rectangular":
                        x1, y1, width, height = selection_data
                        # Ajustar coordenadas según el zoom y pan
                        x1 = int(x1 * self.zoom_factor) + x
                        y1 = int(y1 * self.zoom_factor) + y
                        width = int(width * self.zoom_factor)
                        height = int(height * self.zoom_factor)
                        painter.setPen(QPen(QColor(255, 0, 0), 2))
                        painter.drawRect(x1, y1, width, height)
                    else:  # circular
                        center, radius = selection_data
                        # Ajustar coordenadas según el zoom y pan
                        center_x = int(center[0] * self.zoom_factor) + x
                        center_y = int(center[1] * self.zoom_factor) + y
                        radius = int(radius * self.zoom_factor)
                        
                        # Dibujar el círculo usando un método alternativo
                        painter.setPen(Qt.PenStyle.NoPen)  # Sin borde
                        painter.setBrush(QColor(255, 0, 0, 30))  # Relleno semi-transparente
                        painter.drawEllipse(QPoint(center_x, center_y), radius, radius)
                        
                        # Dibujar el borde del círculo
                        painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                        painter.setBrush(Qt.BrushStyle.NoBrush)  # Sin relleno
                        painter.drawEllipse(QPoint(center_x, center_y), radius, radius)
                
                # Dibujar selección en progreso
                if self.selection_tool.is_drawing and self.selection_tool.start_point and self.selection_tool.end_point:
                    if self.selection_tool.selection_type == "rectangular":
                        x1, y1 = self.selection_tool.start_point
                        x2, y2 = self.selection_tool.end_point
                        # Ajustar coordenadas según el zoom y pan
                        x1 = int(x1 * self.zoom_factor) + x
                        y1 = int(y1 * self.zoom_factor) + y
                        x2 = int(x2 * self.zoom_factor) + x
                        y2 = int(y2 * self.zoom_factor) + y
                        width = abs(x2 - x1)
                        height = abs(y2 - y1)
                        painter.setPen(QPen(QColor(255, 0, 0), 2))
                        painter.drawRect(min(x1, x2), min(y1, y2), width, height)
                    else:  # circular
                        center = self.selection_tool.start_point
                        end = self.selection_tool.end_point
                        # Calcular radio en coordenadas originales
                        dx = end[0] - center[0]
                        dy = end[1] - center[1]
                        radius = int(((dx * dx + dy * dy) ** 0.5) * self.zoom_factor)
                        
                        # Ajustar coordenadas según el zoom y pan
                        center_x = int(center[0] * self.zoom_factor) + x
                        center_y = int(center[1] * self.zoom_factor) + y
                        
                        # Dibujar el círculo usando un método alternativo
                        painter.setPen(Qt.PenStyle.NoPen)  # Sin borde
                        painter.setBrush(QColor(255, 0, 0, 30))  # Relleno semi-transparente
                        painter.drawEllipse(QPoint(center_x, center_y), radius, radius)
                        
                        # Dibujar el borde del círculo
                        painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                        painter.setBrush(Qt.BrushStyle.NoBrush)  # Sin relleno
                        painter.drawEllipse(QPoint(center_x, center_y), radius, radius)
            
            painter.end()
            
            # Mostrar la imagen
            self.image_label.setPixmap(final_pixmap)
            
        except Exception as e:
            self.statusBar.showMessage(f"Error al actualizar la imagen: {str(e)}")
            traceback.print_exc()

    def save_current_selection(self):
        """Guardar la selección actual como archivo DICOM"""
        if self.current_image is None:
            QMessageBox.warning(self, "Error", "No hay imagen cargada")
            return
            
        if not self.selection_tool.final_selection:
            QMessageBox.warning(self, "Error", "No hay selección activa para guardar")
            return
            
        try:
            # Obtener el slice actual si es una imagen 3D
            if len(self.current_image.shape) == 3:
                current_slice = self.current_image[self.current_slice]
            else:
                current_slice = self.current_image
                
            # Extraer la subimagen
            subimage = self.selection_tool.extract_selection(current_slice)
            if subimage is not None:
                # Crear un nuevo dataset DICOM
                ds = pydicom.Dataset()
                
                # Configurar File Meta Information
                ds.file_meta = pydicom.dataset.FileMetaDataset()
                ds.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage
                ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
                ds.file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
                ds.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'  # Implicit VR Little Endian
                
                # Configurar metadatos básicos
                ds.SpecificCharacterSet = 'ISO_IR 192'
                ds.ImageType = ['DERIVED', 'SECONDARY']
                ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
                ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
                ds.StudyDate = pydicom.valuerep.DA(datetime.now().strftime('%Y%m%d'))
                ds.StudyTime = pydicom.valuerep.TM(datetime.now().strftime('%H%M%S'))
                ds.AccessionNumber = ''
                ds.Modality = 'OT'  # Other
                ds.Manufacturer = 'DICOM Viewer'
                ds.SeriesDescription = 'Selección de imagen'
                
                # Si hay metadatos originales, intentar copiar algunos campos relevantes
                if self.current_dicom_metadata is not None:
                    try:
                        for tag in ['PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
                                  'StudyInstanceUID', 'StudyID', 'SeriesInstanceUID', 'SeriesNumber']:
                            if hasattr(self.current_dicom_metadata, tag):
                                setattr(ds, tag, getattr(self.current_dicom_metadata, tag))
                    except Exception as e:
                        print(f"Error al copiar metadatos originales: {str(e)}")
                
                # Actualizar dimensiones y datos de píxeles
                ds.Rows = subimage.shape[0]
                ds.Columns = subimage.shape[1]
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.PixelRepresentation = 0
                ds.BitsAllocated = 8
                ds.BitsStored = 8
                ds.HighBit = 7
                
                # Convertir la imagen al formato correcto si es necesario
                if len(subimage.shape) == 3:  # Si es RGB, convertir a escala de grises
                    subimage = cv2.cvtColor(subimage, cv2.COLOR_RGB2GRAY)
                
                if subimage.dtype != np.uint8:
                    subimage = ((subimage - np.min(subimage)) * 255 / 
                              (np.max(subimage) - np.min(subimage))).astype(np.uint8)
                
                # Agregar los datos de píxeles
                ds.PixelData = subimage.tobytes()
                
                # Generar nombre de archivo único con timestamp
                current_date = datetime.now()
                date_folder = current_date.strftime("%Y-%m-%d")
                timestamp = current_date.strftime("%H%M%S")
                
                # Crear carpeta para la fecha si no existe
                date_path = os.path.join(self.default_save_path, date_folder)
                if not os.path.exists(date_path):
                    os.makedirs(date_path)
                
                # Generar nombre de archivo y ruta completa
                filename = f"selection_{timestamp}.dcm"
                filepath = os.path.join(date_path, filename)
                
                # Guardar el archivo DICOM
                ds.save_as(filepath, write_like_original=False)
                
                self.statusBar.showMessage(f"Selección guardada en: {filepath}")
                QMessageBox.information(self, "Éxito", 
                                      f"Selección guardada correctamente en:\n{filepath}")
                
        except Exception as e:
            self.statusBar.showMessage(f"Error al guardar selección: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error al guardar selección:\n{str(e)}")

    def show_selections(self):
        """Abrir el explorador de archivos en la carpeta de selecciones del día actual"""
        try:
            # Obtener la ruta de la carpeta del día actual
            current_date = datetime.now().strftime("%Y-%m-%d")
            date_path = os.path.join(self.default_save_path, current_date)
            
            # Verificar si la carpeta existe
            if not os.path.exists(date_path):
                QMessageBox.warning(self, "Advertencia", 
                                  f"No hay selecciones guardadas para la fecha {current_date}")
                return
            
            # Abrir el explorador de archivos según el sistema operativo
            if os.name == 'nt':  # Windows
                os.startfile(date_path)
            elif os.name == 'posix':  # Linux/Mac
                if sys.platform == 'darwin':  # Mac
                    subprocess.run(['open', date_path])
                else:  # Linux
                    subprocess.run(['xdg-open', date_path])
            
            self.statusBar.showMessage(f"Explorador abierto en: {date_path}")
            
        except Exception as e:
            self.statusBar.showMessage(f"Error al abrir el explorador: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error al abrir el explorador:\n{str(e)}")

    def download_selections(self, viewer):
        """Descargar las selecciones guardadas en formato DICOM"""
        try:
            # Obtener la ruta de destino
            save_dir = QFileDialog.getExistingDirectory(
                viewer,
                "Seleccionar Directorio de Destino",
                "",
                QFileDialog.Option.ShowDirsOnly
            )
            
            if not save_dir:
                return
                
            # Obtener la fecha actual
            current_date = datetime.now().strftime("%d/%m/%Y")
            
            # Crear un diálogo de progreso
            progress = QProgressDialog("Descargando selecciones...", "Cancelar", 0, len(self.selection_sequence), viewer)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            
            # Guardar cada selección
            for i, selection_id in enumerate(self.selection_sequence):
                if progress.wasCanceled():
                    break
                    
                ds = self.selected_subimages[selection_id]
                
                # Crear un nombre de archivo único con el consecutivo y la fecha
                filename = f"seleccion_{i+1}_{current_date.replace('/', '_')}.dcm"
                filepath = os.path.join(save_dir, filename)
                
                # Guardar el archivo DICOM
                ds.save_as(filepath)
                
                progress.setValue(i + 1)
                QApplication.processEvents()
            
            progress.close()
            
            QMessageBox.information(viewer, "Éxito", "Selecciones descargadas correctamente")
            
        except Exception as e:
            QMessageBox.critical(viewer, "Error", f"Error al descargar selecciones: {str(e)}")
            viewer.statusBar.showMessage(f"Error al descargar selecciones: {str(e)}")

    def load_dicom_file(self, item):
        """Cargar un archivo DICOM desde la lista de archivos"""
        try:
            # Obtener la ruta completa del archivo
            if isinstance(item, QListWidgetItem):
                file_name = item.text()
                # Buscar la ruta completa en la lista de archivos
                file_path = None
                for full_path in self.current_directory_files:
                    if os.path.basename(full_path) == file_name:
                        file_path = full_path
                        break
            else:
                file_path = item

            if not file_path or not os.path.exists(file_path):
                QMessageBox.warning(self, "Error", f"El archivo {file_path} no existe")
                return

            # Mostrar indicador de carga
            self.statusBar.showMessage("Cargando archivo...")
            QApplication.processEvents()

            # Leer archivo DICOM de manera asíncrona
            try:
                dicom = pydicom.dcmread(file_path, force=True)
            except Exception as e:
                raise Exception(f"Error al leer archivo DICOM: {str(e)}")
            
            # Verificar y obtener los datos de píxeles
            if not hasattr(dicom, 'pixel_array'):
                raise Exception("El archivo no contiene datos de imagen")
            
            # Procesar la imagen de manera eficiente
            try:
                if len(dicom.pixel_array.shape) == 3:
                    # Para stacks 3D, procesar solo el primer slice inicialmente
                    self.original_image = self.process_dicom_image(dicom.pixel_array[0], dicom)
                    self.current_image = self.original_image.copy()
                else:
                    # Para imágenes 2D
                    self.original_image = self.process_dicom_image(dicom.pixel_array, dicom)
                    self.current_image = self.original_image.copy()
            except Exception as e:
                raise Exception(f"Error al procesar imagen: {str(e)}")
            
            # Actualizar el índice del archivo actual
            if isinstance(item, QListWidgetItem):
                self.current_file_index = self.file_list.row(item)
            else:
                self.current_file_index = self.current_directory_files.index(file_path)
            
            # Actualizar UI
            self.slice_slider.setValue(self.current_file_index)
            self.slice_label.setText(f"Archivo: {self.current_file_index + 1}/{len(self.current_directory_files)}")
            
            # Guardar metadata para acceso en tiempo real
            self.current_dicom_metadata = dicom
            
            # Mostrar metadata e imagen
            self.display_metadata(dicom)
            self.display_image()
            
            self.statusBar.showMessage(f"Archivo DICOM cargado: {os.path.basename(file_path)}")
            
        except Exception as e:
            self.statusBar.showMessage(f"Error al cargar DICOM: {str(e)}")
            QMessageBox.critical(self, "Error", f"No se pudo cargar el archivo:\n{str(e)}")

    def load_dicom_directory(self):
        """Cargar un directorio de archivos DICOM de manera optimizada usando procesamiento en paralelo"""
        try:
            # Abrir diálogo para seleccionar directorio
            directory = QFileDialog.getExistingDirectory(
                self, 
                "Seleccionar Directorio DICOM", 
                "", 
                QFileDialog.Option.ShowDirsOnly
            )
            
            if not directory:
                return
                
            # Limpiar lista actual
            self.file_list.clear()
            self.current_directory_files = []
            
            # Configurar diálogo de progreso
            progress = QProgressDialog("Cargando archivos DICOM...", "Cancelar", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setValue(0)
            
            # Obtener lista de todos los archivos en el directorio
            all_files = [f for f in os.listdir(directory) 
                        if os.path.isfile(os.path.join(directory, f))]
            
            if not all_files:
                QMessageBox.warning(self, "Advertencia", 
                                  "No se encontraron archivos en el directorio seleccionado")
                return
            
            # Configurar procesamiento en paralelo
            num_cores = max(1, multiprocessing.cpu_count() - 1)  # Dejar un núcleo libre
            chunk_size = max(1, len(all_files) // (num_cores * 4))  # Dividir en chunks más pequeños
            
            # Dividir archivos en chunks y preparar datos para procesamiento
            chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
            chunk_data = [(chunk, directory) for chunk in chunks]
            total_chunks = len(chunks)
            
            # Procesar chunks en paralelo
            valid_files = []
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = [executor.submit(process_chunk, data) for data in chunk_data]
                
                for i, future in enumerate(as_completed(futures)):
                    if progress.wasCanceled():
                        executor.shutdown(wait=False)
                        return
                    
                    try:
                        chunk_valid_files = future.result()
                        valid_files.extend(chunk_valid_files)
                        
                        # Actualizar progreso
                        progress_value = min(100, int((i + 1) * 100 / total_chunks))
                        progress.setValue(progress_value)
                        progress.setLabelText(f"Procesando archivos... ({len(valid_files)} válidos encontrados)")
                        QApplication.processEvents()
                        
                    except Exception as e:
                        print(f"Error procesando chunk: {str(e)}")
                        continue
            
            # Ordenar archivos por nombre para mantener secuencia
            valid_files.sort()
            
            # Guardar la lista de archivos válidos con rutas completas
            self.current_directory_files = valid_files
            self.current_file_index = 0
            
            # Agregar archivos válidos a la lista mostrando solo el nombre del archivo
            self.file_list.addItems([os.path.basename(f) for f in valid_files])
            
            progress.close()
            
            if self.file_list.count() == 0:
                QMessageBox.warning(self, "Advertencia", 
                                  "No se encontraron archivos DICOM válidos en el directorio seleccionado")
            else:
                # Configurar el slider
                self.slice_slider.setMinimum(0)
                self.slice_slider.setMaximum(len(valid_files) - 1)
                self.slice_slider.setValue(0)
                self.slice_label.setText(f"Archivo: 1/{len(valid_files)}")
                
                # Habilitar controles de navegación
                self.prev_slice_button.setEnabled(True)
                self.next_slice_button.setEnabled(True)
                self.play_button.setEnabled(True)
                self.speed_selector.setEnabled(True)
                self.slice_slider.setEnabled(True)
                
                # Habilitar controles de filtros
                self.filters_combo.setEnabled(True)
                self.reset_filters_button.setEnabled(True)
                
                # Habilitar botón de modelo predictivo
                self.predictive_model_button.setEnabled(True)
                
                # Cargar el primer archivo
                self.load_dicom_file(self.current_directory_files[0])
                
                self.statusBar.showMessage(f"Directorio cargado: {self.file_list.count()} archivos DICOM encontrados")
                
        except Exception as e:
            self.statusBar.showMessage(f"Error al cargar directorio: {str(e)}")
            QMessageBox.critical(self, "Error", 
                               f"No se pudo cargar el directorio:\n{str(e)}")

    def clear_selections(self):
        """Limpiar todas las selecciones actuales"""
        self.selection_tool.start_point = None
        self.selection_tool.end_point = None
        self.selection_tool.is_drawing = False
        self.selection_tool.final_selection = None
        self.update_image_with_selection()

    def deferred_update(self):
        """Actualización diferida de la interfaz"""
        if self.pending_update:
            try:
                self.update_image_with_selection()
            except Exception as e:
                self.statusBar.showMessage(f"Error en actualización: {str(e)}")
            finally:
                self.pending_update = False
    
    def request_update(self):
        """Solicitar una actualización diferida"""
        if not self.pending_update:
            self.pending_update = True
            self.update_timer.start(16)  # ~60 FPS

    def display_image(self):
        """Mostrar la imagen actual con zoom y pan"""
        if self.current_image is None:
            return
            
        try:
            # Validar dimensiones de la imagen
            if len(self.current_image.shape) not in [2, 3, 4]:
                raise ValueError(f"Dimensiones de imagen no soportadas: {self.current_image.shape}")
                
            # Obtener el corte actual
            if len(self.current_image.shape) == 4:
                if self.current_slice >= self.current_image.shape[0]:
                    self.current_slice = 0
                slice_data = self.current_image[self.current_slice].copy()
                # Actualizar imagen original para filtros si es necesario
                if not hasattr(self, 'original_image') or self.original_image.shape != self.current_image.shape:
                    self.original_image = self.current_image.copy()
            elif len(self.current_image.shape) == 3:
                if self.current_image.shape[-1] == 3:
                    slice_data = self.current_image.copy()
                else:
                    if self.current_slice >= self.current_image.shape[0]:
                        self.current_slice = 0
                    slice_data = self.current_image[self.current_slice].copy()
                    # Actualizar imagen original para filtros si es necesario
                    if not hasattr(self, 'original_image') or self.original_image.shape != self.current_image.shape:
                        self.original_image = self.current_image.copy()
            else:
                slice_data = self.current_image.copy()
                # Actualizar imagen original para filtros si es necesario
                if not hasattr(self, 'original_image') or self.original_image.shape != self.current_image.shape:
                    self.original_image = self.current_image.copy()
            
            # Validar datos de la imagen
            if slice_data.size == 0:
                raise ValueError("La imagen está vacía")
                
            if not np.all(np.isfinite(slice_data)):
                raise ValueError("La imagen contiene valores no válidos (inf o nan)")
            
            # Asegurar que los datos sean contiguos
            if not slice_data.flags['C_CONTIGUOUS']:
                slice_data = np.ascontiguousarray(slice_data)
            
            # Manejar las dimensiones
            shape = slice_data.shape
            
            if len(shape) == 3 and shape[2] == 3:
                height, width, _ = shape
                bytes_per_line = width * 3
                qimage_format = QImage.Format.Format_RGB888
            elif len(shape) == 2:
                height, width = shape
                bytes_per_line = width
                qimage_format = QImage.Format.Format_Grayscale8
            else:
                raise ValueError(f"Dimensiones no soportadas después del procesamiento: {shape}")
            
            # Crear QImage
            image = QImage(slice_data.data, width, height, 
                         bytes_per_line, qimage_format)
            
            # Convertir a QPixmap
            pixmap = QPixmap.fromImage(image)
            
            # Obtener dimensiones del label
            label_size = self.image_label.size()
            
            # Calcular el tamaño de la imagen con zoom
            zoomed_size = QSize(
                int(width * self.zoom_factor),
                int(height * self.zoom_factor)
            )
            
            # Escalar la imagen con zoom manteniendo la calidad
            scaled_pixmap = pixmap.scaled(
                zoomed_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Crear un nuevo pixmap del tamaño del label
            final_pixmap = QPixmap(label_size)
            final_pixmap.fill(Qt.GlobalColor.transparent)
            
            # Calcular la posición de la imagen con pan
            x = int((label_size.width() - scaled_pixmap.width()) // 2 + self.pan_offset.x())
            y = int((label_size.height() - scaled_pixmap.height()) // 2 + self.pan_offset.y())
            
            # Dibujar la imagen escalada en la posición correcta
            painter = QPainter(final_pixmap)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawPixmap(x, y, scaled_pixmap)
            
            # Si hay una selección activa, dibujarla
            if self.is_selection_active():
                # Configurar el renderizado para círculos suaves
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setRenderHint(QPainter.RenderHint.HighQualityAntialiasing)
                
                # Dibujar selección existente
                if self.selection_tool.final_selection:
                    selection_type, selection_data = self.selection_tool.final_selection
                    if selection_type == "rectangular":
                        x1, y1, width, height = selection_data
                        # Ajustar coordenadas según el zoom y pan
                        x1 = int(x1 * self.zoom_factor) + x
                        y1 = int(y1 * self.zoom_factor) + y
                        width = int(width * self.zoom_factor)
                        height = int(height * self.zoom_factor)
                        painter.setPen(QPen(QColor(255, 0, 0), 2))
                        painter.drawRect(x1, y1, width, height)
                    else:  # circular
                        center, radius = selection_data
                        # Ajustar coordenadas según el zoom y pan
                        center_x = int(center[0] * self.zoom_factor) + x
                        center_y = int(center[1] * self.zoom_factor) + y
                        radius = int(radius * self.zoom_factor)
                        
                        # Dibujar el círculo usando un método alternativo
                        painter.setPen(Qt.PenStyle.NoPen)  # Sin borde
                        painter.setBrush(QColor(255, 0, 0, 30))  # Relleno semi-transparente
                        painter.drawEllipse(QPoint(center_x, center_y), radius, radius)
                        
                        # Dibujar el borde del círculo
                        painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                        painter.setBrush(Qt.BrushStyle.NoBrush)  # Sin relleno
                        painter.drawEllipse(QPoint(center_x, center_y), radius, radius)
                
                # Dibujar selección en progreso
                if self.selection_tool.is_drawing and self.selection_tool.start_point and self.selection_tool.end_point:
                    if self.selection_tool.selection_type == "rectangular":
                        x1, y1 = self.selection_tool.start_point
                        x2, y2 = self.selection_tool.end_point
                        # Ajustar coordenadas según el zoom y pan
                        x1 = int(x1 * self.zoom_factor) + x
                        y1 = int(y1 * self.zoom_factor) + y
                        x2 = int(x2 * self.zoom_factor) + x
                        y2 = int(y2 * self.zoom_factor) + y
                        width = abs(x2 - x1)
                        height = abs(y2 - y1)
                        painter.setPen(QPen(QColor(255, 0, 0), 2))
                        painter.drawRect(min(x1, x2), min(y1, y2), width, height)
                    else:  # circular
                        center = self.selection_tool.start_point
                        end = self.selection_tool.end_point
                        # Calcular radio en coordenadas originales
                        dx = end[0] - center[0]
                        dy = end[1] - center[1]
                        radius = int(((dx * dx + dy * dy) ** 0.5) * self.zoom_factor)
                        
                        # Ajustar coordenadas según el zoom y pan
                        center_x = int(center[0] * self.zoom_factor) + x
                        center_y = int(center[1] * self.zoom_factor) + y
                        
                        # Dibujar el círculo usando un método alternativo
                        painter.setPen(Qt.PenStyle.NoPen)  # Sin borde
                        painter.setBrush(QColor(255, 0, 0, 30))  # Relleno semi-transparente
                        painter.drawEllipse(QPoint(center_x, center_y), radius, radius)
                        
                        # Dibujar el borde del círculo
                        painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                        painter.setBrush(Qt.BrushStyle.NoBrush)  # Sin relleno
                        painter.drawEllipse(QPoint(center_x, center_y), radius, radius)
            
            painter.end()
            
            # Mostrar la imagen
            self.image_label.setPixmap(final_pixmap)
            
            # Actualizar información de slice
            if len(self.current_image.shape) == 4:
                self.total_slices = self.current_image.shape[0]
                self.slice_label.setText(f"Corte: {self.current_slice + 1}/{self.total_slices}")
            
            # Actualizar información inicial de la imagen
            self.image_info_label.setText("Cargando imagen...")
            
            # Actualizar información de coordenadas con la posición actual del mouse
            if hasattr(self, 'image_label') and self.image_label.underMouse():
                mouse_pos = self.image_label.mapFromGlobal(self.cursor().pos())
                self.update_coordinates_display(mouse_pos)
            
            # Actualizar barra de estado
            self.statusBar.showMessage(
                f"Mostrando imagen: {width}x{height} píxeles | Zoom: {self.zoom_factor:.1f}x"
            )
            
        except Exception as e:
            self.statusBar.showMessage(f"Error al mostrar la imagen: {str(e)}")
            traceback.print_exc()
        
        # Posicionar el cuadro de información en la esquina inferior derecha
        self.position_image_info_label()

    def position_image_info_label(self):
        """Posicionar el cuadro de información en la esquina inferior derecha del canvas"""
        if hasattr(self, 'image_info_label') and hasattr(self, 'image_label'):
            # Obtener el tamaño del canvas de la imagen
            canvas_size = self.image_label.size()
            
            # Calcular la posición para la esquina inferior derecha
            # Dejar un margen de 10 píxeles desde los bordes
            x = canvas_size.width() - self.image_info_label.width() - 10
            y = canvas_size.height() - self.image_info_label.height() - 10
            
            # Asegurar que no se salga de los límites
            x = max(0, x)
            y = max(0, y)
            
            # Posicionar el widget
            self.image_info_label.move(x, y)

    def apply_image_filter(self, filter_name):
        """Aplicar filtro de imagen seleccionado"""
        if self.current_image is None:
            return
        
        try:
            # Crear una copia de la imagen original para aplicar filtros
            if not hasattr(self, 'original_image'):
                self.original_image = self.current_image.copy()
            
            # Aplicar el filtro seleccionado
            if filter_name == "Sin filtro":
                # Restaurar imagen original
                self.current_image = self.original_image.copy()
            elif filter_name == "Blur":
                # Aplicar filtro de desenfoque gaussiano
                if len(self.current_image.shape) == 3:
                    # Para imágenes RGB, aplicar a cada canal
                    self.current_image = cv2.GaussianBlur(self.original_image, (15, 15), 0)
                else:
                    # Para imágenes en escala de grises
                    self.current_image = cv2.GaussianBlur(self.original_image, (15, 15), 0)
            elif filter_name == "Edge":
                # Aplicar detección de bordes con Canny
                if len(self.original_image.shape) == 3:
                    gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = self.original_image
                edges = cv2.Canny(gray, 50, 150)
                self.current_image = edges
            elif filter_name == "Sharpness":
                # Aplicar filtro de nitidez (kernel de convolución)
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                if len(self.original_image.shape) == 3:
                    # Para imágenes RGB, aplicar a cada canal
                    self.current_image = cv2.filter2D(self.original_image, -1, kernel)
                else:
                    # Para imágenes en escala de grises
                    self.current_image = cv2.filter2D(self.original_image, -1, kernel)
            elif filter_name == "Full Dynamic":
                # Aplicar estiramiento de histograma completo
                if len(self.original_image.shape) == 3:
                    # Para imágenes RGB, convertir a HSV y estirar el canal V
                    hsv = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2HSV)
                    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
                    self.current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                else:
                    # Para imágenes en escala de grises
                    self.current_image = cv2.equalizeHist(self.original_image)
            elif filter_name == "Skull":
                # Aplicar filtro específico para cráneo (realce de bordes óseos)
                if len(self.original_image.shape) == 3:
                    gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = self.original_image
                
                # Aplicar filtro de realce de bordes
                kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
                enhanced = cv2.filter2D(gray, -1, kernel)
                
                # Combinar con la imagen original para realzar bordes
                self.current_image = cv2.addWeighted(gray, 0.7, enhanced, 0.3, 0)
            
            # Redibujar la imagen con el filtro aplicado
            self.display_image()
            
            # Mostrar mensaje de confirmación
            self.statusBar.showMessage(f"Filtro aplicado: {filter_name}")
            
        except Exception as e:
            self.statusBar.showMessage(f"Error al aplicar filtro: {str(e)}")
            traceback.print_exc()

    def reset_filters(self):
        """Resetear todos los filtros y restaurar imagen original"""
        if self.current_image is None or not hasattr(self, 'original_image'):
            return
        
        try:
            # Restaurar imagen original
            self.current_image = self.original_image.copy()
            
            # Resetear el combo box a "Sin filtro"
            self.filters_combo.setCurrentText("Sin filtro")
            
            # Redibujar la imagen
            self.display_image()
            
            # Mostrar mensaje de confirmación
            self.statusBar.showMessage("Filtros reseteados - Imagen original restaurada")
            
        except Exception as e:
            self.statusBar.showMessage(f"Error al resetear filtros: {str(e)}")
            traceback.print_exc()

    def execute_predictive_model(self):
        """Ejecutar modelo predictivo sobre la imagen actual usando HeartSegmentation.predict()"""
        if self.current_image is None:
            QMessageBox.warning(self, "Error", "No hay imagen cargada para analizar")
            return
        
        try:
            # Mostrar diálogo de progreso
            progress = QProgressDialog("Ejecutando modelo predictivo...", "Cancelar", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setValue(10)
            
            # Preparar la imagen para el modelo (siguiendo el formato de HeartSegmentation)
            if len(self.current_image.shape) == 3:
                # Si es una imagen RGB, convertir a escala de grises
                if self.current_image.shape[-1] == 3:
                    image_for_model = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
                else:
                    # Si es un stack 3D, tomar el slice actual
                    image_for_model = self.current_image[self.current_slice] if self.current_slice < self.current_image.shape[0] else self.current_image[0]
            else:
                image_for_model = self.current_image.copy()
            
            progress.setValue(30)
            
            # Normalizar la imagen como lo hace HeartSegmentation
            image_for_model = np.expand_dims(image_for_model, axis=-1)
            image_for_model = image_for_model / np.max(image_for_model) * 255.0
            x = image_for_model / 255.0
            x = np.concatenate([x, x, x], axis=-1)  # Convertir a RGB
            x = np.expand_dims(x, axis=0)  # Agregar dimensión de batch
            
            progress.setValue(50)
            
            # Cargar el modelo como lo hace HeartSegmentation
            from HeartSegmentation import iou, dice_coef, dice_loss
            from tensorflow.keras.utils import CustomObjectScope
            
            model_path = os.path.join(os.path.dirname(__file__), "..", "processing", "model.h5")
            if not os.path.exists(model_path):
                QMessageBox.critical(self, "Error", f"No se encontró el modelo en: {model_path}")
                return
            
            progress.setValue(70)
            
            # Cargar el modelo con los objetos personalizados
            with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
                model = keras.models.load_model(model_path)
            
            progress.setValue(80)
            
            # Realizar predicción usando el mismo método que HeartSegmentation
            mask = model.predict(x)[0]
            mask = mask > 0.5
            mask = mask.astype(np.int32)
            mask = mask * 255
            
            progress.setValue(90)
            
            # Calcular métricas básicas
            mask_area = np.sum(mask > 0)
            total_area = mask.shape[0] * mask.shape[1]
            coverage_percentage = (mask_area / total_area) * 100
            
            # Determinar resultado basado en la cobertura
            if coverage_percentage < 5:
                predicted_class = "Normal"
                confidence_score = 0.95
            elif coverage_percentage < 15:
                predicted_class = "Leve"
                confidence_score = 0.85
            elif coverage_percentage < 30:
                predicted_class = "Moderado"
                confidence_score = 0.75
            else:
                predicted_class = "Severo"
                confidence_score = 0.90
            
            progress.setValue(100)
            
            # Mostrar resultados
            result_message = f"""Resultados del Modelo Predictivo (HeartSegmentation):
            
Clasificación: {predicted_class}
Confianza: {confidence_score:.2%}
Cobertura de Segmentación: {coverage_percentage:.1f}%
Área Segmentada: {mask_area} píxeles
Área Total: {total_area} píxeles

El modelo ha realizado segmentación del corazón usando deep learning."""
            
            # Mostrar resultados en un diálogo
            QMessageBox.information(self, "Resultados del Modelo Predictivo", result_message)
            
            # Actualizar barra de estado
            self.statusBar.showMessage(f"Modelo predictivo ejecutado - Clasificación: {predicted_class} ({confidence_score:.1%})")
            
        except Exception as e:
            self.statusBar.showMessage(f"Error al ejecutar modelo predictivo: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error al ejecutar modelo predictivo:\n{str(e)}")
            traceback.print_exc()

    def change_playback_speed(self, speed_text):
        """Change the playback speed based on selection"""
        if self.current_image is not None and len(self.current_image.shape) == 3:
            self.play_button.setEnabled(True)
            
        if self.is_playing:
            self.play_timer.stop()
            if speed_text == "2 fps":
                self.play_timer.start(500)
            elif speed_text == "1 fps":
                self.play_timer.start(1000)
            elif speed_text == "0.5 fps":
                self.pause_playback()
            
    def next_slice(self):
        """Mostrar siguiente slice o archivo"""
        if not self.current_directory_files:
            return

        try:
            # Deshabilitar controles temporalmente
            self.prev_slice_button.setEnabled(False)
            self.next_slice_button.setEnabled(False)
            self.slice_slider.setEnabled(False)
            QApplication.processEvents()

            # Calcular siguiente índice
            if self.current_file_index < len(self.current_directory_files) - 1:
                self.current_file_index += 1
            else:
                self.current_file_index = 0
            
            # Actualizar UI
            self.slice_slider.setValue(self.current_file_index)
            self.slice_label.setText(f"Archivo: {self.current_file_index + 1}/{len(self.current_directory_files)}")
            QApplication.processEvents()
            
            # Cargar el archivo
            self.load_dicom_file(self.current_directory_files[self.current_file_index])
            
        except Exception as e:
            self.statusBar.showMessage(f"Error al navegar: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error al navegar al siguiente archivo:\n{str(e)}")
        finally:
            # Rehabilitar controles
            self.prev_slice_button.setEnabled(True)
            self.next_slice_button.setEnabled(True)
            self.slice_slider.setEnabled(True)
            self.filters_combo.setEnabled(True)
            self.reset_filters_button.setEnabled(True)
            self.predictive_model_button.setEnabled(True)

    def previous_slice(self):
        """Mostrar slice o archivo anterior"""
        if not self.current_directory_files:
            return

        try:
            # Deshabilitar controles temporalmente
            self.prev_slice_button.setEnabled(False)
            self.next_slice_button.setEnabled(False)
            self.slice_slider.setEnabled(False)
            QApplication.processEvents()

            # Calcular índice anterior
            if self.current_file_index > 0:
                self.current_file_index -= 1
            else:
                self.current_file_index = len(self.current_directory_files) - 1
            
            # Actualizar UI
            self.slice_slider.setValue(self.current_file_index)
            self.slice_label.setText(f"Archivo: {self.current_file_index + 1}/{len(self.current_directory_files)}")
            QApplication.processEvents()
            
            # Cargar el archivo
            self.load_dicom_file(self.current_directory_files[self.current_file_index])
            
            
        except Exception as e:
            self.statusBar.showMessage(f"Error al navegar: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error al navegar al archivo anterior:\n{str(e)}")
        finally:
            # Rehabilitar controles
            self.prev_slice_button.setEnabled(True)
            self.next_slice_button.setEnabled(True)
            self.slice_slider.setEnabled(True)
            self.filters_combo.setEnabled(True)
            self.reset_filters_button.setEnabled(True)
            self.predictive_model_button.setEnabled(True)

    def apply_filter(self, filter_name):
        """Aplicar filtro seleccionado a todas las imágenes"""
        if self.current_image is None:
            return
            
        try:
            # Mostrar diálogo de progreso
            progress = QProgressDialog("Aplicando filtro...", "Cancelar", 0, 
                                     self.total_slices, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            
            # Crear copia del stack completo de manera eficiente
            if len(self.current_image.shape) == 3:
                filtered_stack = self.original_image.copy()
            else:
                filtered_stack = self.original_image.copy()[np.newaxis, ...]
            
            # Preasignar arrays para operaciones vectorizadas
            temp_buffer = np.empty_like(filtered_stack[0], dtype=np.float32)
            
            # Aplicar filtro a cada slice
            for i in range(self.total_slices):
                if progress.wasCanceled():
                    self.current_image = self.original_image.copy()
                    self.display_image()
                    return
                
                # Obtener slice actual
                slice_data = filtered_stack[i]
                
                # Aplicar el filtro seleccionado de manera optimizada
                if filter_name == "Ninguno":
                    self.current_image = self.original_image.copy()
                    break
                    
                elif filter_name == "Ajuste Ventana/Nivel":
                    p2, p98 = np.percentile(slice_data, (2, 98))
                    np.clip(slice_data, p2, p98, out=slice_data)
                    
                elif filter_name == "Ventana Ósea":
                    window_center = 400
                    window_width = 1800
                    np.clip(slice_data, window_center - window_width/2, 
                           window_center + window_width/2, out=slice_data)
                    
                elif filter_name == "Ventana Pulmonar":
                    window_center = -600
                    window_width = 1500
                    np.clip(slice_data, window_center - window_width/2, 
                           window_center + window_width/2, out=slice_data)
                    
                elif filter_name == "Ventana Cerebral":
                    window_center = 40
                    window_width = 80
                    np.clip(slice_data, window_center - window_width/2, 
                           window_center + window_width/2, out=slice_data)
                    
                elif filter_name == "Realce de Bordes":
                    dx = cv2.Sobel(slice_data, cv2.CV_32F, 1, 0, ksize=3)
                    dy = cv2.Sobel(slice_data, cv2.CV_32F, 0, 1, ksize=3)
                    np.sqrt(dx**2 + dy**2, out=slice_data)
                    
                elif filter_name == "Reducción de Ruido":
                    cv2.GaussianBlur(slice_data, (5, 5), 0, dst=slice_data)
                    
                elif filter_name == "Contraste Local":
                    np.subtract(slice_data, np.min(slice_data), out=temp_buffer)
                    if np.max(slice_data) != np.min(slice_data):
                        np.divide(temp_buffer, np.max(slice_data) - np.min(slice_data), out=temp_buffer)
                    np.multiply(temp_buffer, 255, out=temp_buffer)
                    temp_buffer = temp_buffer.astype(np.uint8)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    slice_data[:] = clahe.apply(temp_buffer)
                    
                elif filter_name == "Invertir":
                    np.subtract(np.max(slice_data), slice_data, out=slice_data)
                    
                elif filter_name == "Realzar Detalles":
                    cv2.GaussianBlur(slice_data, (5, 5), 1.0, dst=temp_buffer)
                    cv2.addWeighted(slice_data, 1.5, temp_buffer, -0.5, 0, dst=slice_data)
                
                # Normalizar la imagen resultante de manera eficiente
                np.subtract(slice_data, np.min(slice_data), out=slice_data)
                if np.max(slice_data) != 0:
                    np.divide(slice_data, np.max(slice_data), out=slice_data)
                    np.multiply(slice_data, 255, out=slice_data)
                slice_data = slice_data.astype(np.uint8)
                
                # Actualizar el stack
                filtered_stack[i] = slice_data
                
                # Actualizar progreso
                progress.setValue(i + 1)
                QApplication.processEvents()
            
            # Actualizar la imagen actual con el stack filtrado
            self.current_image = filtered_stack
            if len(self.current_image.shape) == 3 and self.current_image.shape[0] == 1:
                self.current_image = self.current_image[0]
            
            # Mostrar la imagen actualizada
            self.display_image()
            self.statusBar.showMessage(f"Filtro {filter_name} aplicado a todas las imágenes")
            
        except Exception as e:
            self.statusBar.showMessage(f"Error al aplicar filtro: {str(e)}")
            # Restaurar imagen original en caso de error
            self.current_image = self.original_image.copy()
            self.display_image()
    
    def analyze_image(self):
        """Analizar la imagen actual usando solo OpenCV"""
        if self.current_image is None:
            return
            
        try:
            # Obtener la imagen actual
            if len(self.current_image.shape) == 3:
                slice_data = self.current_image[self.current_slice].copy()
            else:
                slice_data = self.current_image.copy()
            
            # Normalizar a uint8 para OpenCV
            if slice_data.dtype != np.uint8:
                slice_data = ((slice_data - np.min(slice_data)) * 255 / 
                            (np.max(slice_data) - np.min(slice_data))).astype(np.uint8)
            
            # Análisis básico de la imagen usando OpenCV
            mean_val = cv2.mean(slice_data)[0]
            std_val = cv2.meanStdDev(slice_data)[1][0][0]
            min_val = np.min(slice_data)
            max_val = np.max(slice_data)
            
            # Detección de bordes usando Canny
            edges = cv2.Canny(slice_data, 50, 150)
            num_edges = np.count_nonzero(edges)
            
            # Análisis de histograma
            hist = cv2.calcHist([slice_data], [0], None, [256], [0, 256])
            peak_intensity = np.argmax(hist)
            
            # Crear mensaje con resultados
            analysis_text = (
                "Análisis de la Imagen:\n\n"
                f"Intensidad Media: {mean_val:.2f}\n"
                f"Desviación Estándar: {std_val:.2f}\n"
                f"Valor Mínimo: {min_val}\n"
                f"Valor Máximo: {max_val}\n"
                f"Pico de Intensidad: {peak_intensity}\n"
                f"Cantidad de Bordes: {num_edges}\n"
                f"Dimensiones: {slice_data.shape}\n"
            )
            
            # Mostrar resultados
            QMessageBox.information(self, "Análisis de Imagen", analysis_text)
            
        except Exception as e:
            self.statusBar.showMessage(f"Error en el análisis de imagen: {str(e)}")
            QMessageBox.critical(self, "Error", 
                               f"Error durante el análisis de la imagen:\n{str(e)}")

    def slider_changed(self, value):
        """Manejar cambios en el slider"""
        if not self.current_directory_files:
            return
            
        try:
            # Deshabilitar controles temporalmente
            self.prev_slice_button.setEnabled(False)
            self.next_slice_button.setEnabled(False)
            self.slice_slider.setEnabled(False)
            QApplication.processEvents()

            self.current_file_index = value
            self.load_dicom_file(self.current_directory_files[self.current_file_index])
            self.slice_label.setText(f"Archivo: {self.current_file_index + 1}/{len(self.current_directory_files)}")
            
        except Exception as e:
            self.statusBar.showMessage(f"Error al cambiar archivo: {str(e)}")
        finally:
            # Rehabilitar controles
            self.prev_slice_button.setEnabled(True)
            self.next_slice_button.setEnabled(True)
            self.slice_slider.setEnabled(True)
            self.filters_combo.setEnabled(True)
            self.reset_filters_button.setEnabled(True)
            self.predictive_model_button.setEnabled(True)

    def process_dicom_image(self, pixel_array, dicom):
        """Procesar una imagen DICOM de manera estandarizada"""
        try:
            # Validar entrada
            if pixel_array is None or pixel_array.size == 0:
                raise ValueError("Array de píxeles vacío o nulo")
            
            # Convertir a float32 de manera eficiente
            if pixel_array.dtype != np.float32:
                pixel_array = pixel_array.astype(np.float32, copy=False)
            
            # Aplicar rescale slope e intercept de manera vectorizada
            slope = float(dicom.get('RescaleSlope', 1))
            intercept = float(dicom.get('RescaleIntercept', 0))
            pixel_array = np.multiply(pixel_array, slope, out=pixel_array)
            pixel_array = np.add(pixel_array, intercept, out=pixel_array)
            
            # Obtener y procesar Window Center y Width
            window_center = dicom.get('WindowCenter', None)
            window_width = dicom.get('WindowWidth', None)
            
            if isinstance(window_center, pydicom.multival.MultiValue):
                window_center = float(window_center[0])
            elif window_center is None:
                window_center = np.mean(pixel_array)
            else:
                window_center = float(window_center)
                
            if isinstance(window_width, pydicom.multival.MultiValue):
                window_width = float(window_width[0])
            elif window_width is None:
                window_width = np.max(pixel_array) - np.min(pixel_array)
            else:
                window_width = float(window_width)
            
            # Aplicar windowing de manera vectorizada
            window_min = window_center - window_width/2
            window_max = window_center + window_width/2
            pixel_array = np.clip(pixel_array, window_min, window_max, out=pixel_array)
            
            # Normalizar y convertir a uint8 de manera eficiente
            pixel_array = np.subtract(pixel_array, window_min, out=pixel_array)
            if window_max != window_min:
                pixel_array = np.divide(pixel_array, window_max - window_min, out=pixel_array)
            pixel_array = np.multiply(pixel_array, 255, out=pixel_array)
            
            return pixel_array.astype(np.uint8, copy=False)
            
        except Exception as e:
            raise Exception(f"Error al procesar imagen: {str(e)}")

    def load_dicom(self):
        """Cargar y mostrar un archivo DICOM"""
        try:
            file_name = QFileDialog.getOpenFileName(
                self, 
                "Abrir archivo DICOM", 
                "", 
                "Todos los archivos (*)")[0]
            
            if not file_name:
                return
                
            # Leer archivo DICOM
            dicom = pydicom.dcmread(file_name, force=True)
            
            # Guardar los metadatos originales
            self.current_dicom_metadata = dicom
            
            # Verificar y obtener los datos de píxeles
            if not hasattr(dicom, 'pixel_array'):
                raise Exception("El archivo no contiene datos de imagen")
            
            # Procesar la imagen
            if len(dicom.pixel_array.shape) == 3:
                # Para stacks 3D
                processed_images = []
                for i in range(dicom.pixel_array.shape[0]):
                    processed = self.process_dicom_image(dicom.pixel_array[i], dicom)
                    processed_images.append(processed)
                self.original_image = np.stack(processed_images)
            else:
                # Para imágenes 2D
                self.original_image = self.process_dicom_image(dicom.pixel_array, dicom)
            
            self.current_image = self.original_image.copy()
            
            # Configurar la navegación según dimensiones
            if len(self.original_image.shape) == 4:  # Para imágenes 4D
                self.total_slices = self.original_image.shape[0]
                self.current_slice = 0
                self.setup_navigation_controls(True)
            elif len(self.original_image.shape) == 3 and self.original_image.shape[-1] != 3:
                self.total_slices = self.original_image.shape[0]
                self.current_slice = 0
                self.setup_navigation_controls(True)
            else:
                self.total_slices = 1
                self.current_slice = 0
                self.setup_navigation_controls(False)
            
            # Limpiar selecciones anteriores
            self.clear_selections()
            
            # Guardar metadata para acceso en tiempo real
            self.current_dicom_metadata = dicom
            
            # Guardar imagen original para filtros
            self.original_image = self.current_image.copy()
            
            # Mostrar metadata e imagen
            self.display_metadata(dicom)
            self.display_image()
            
            self.statusBar.showMessage(f"Archivo DICOM cargado: {os.path.basename(file_name)}")
            
        except Exception as e:
            self.statusBar.showMessage(f"Error al cargar DICOM: {str(e)}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", 
                               f"No se pudo cargar el archivo:\n{str(e)}")

    def display_metadata(self, dicom):
        """Mostrar la metadata del archivo DICOM"""
        try:
            metadata_text = "<style>table {width: 100%;} td {padding: 5px; border-bottom: 1px solid #ddd;}</style>"
            metadata_text += "<table>"
            
            # Información básica del archivo
            metadata_text += "<tr><td colspan='2'><h3>Información General</h3></td></tr>"
            
            # Diccionario de campos relevantes y sus etiquetas en español
            relevant_fields = {
                'PatientName': 'Nombre del Paciente',
                'PatientID': 'ID del Paciente',
                'PatientBirthDate': 'Fecha de Nacimiento',
                'PatientSex': 'Sexo',
                'StudyDate': 'Fecha del Estudio',
                'StudyTime': 'Hora del Estudio',
                'StudyDescription': 'Descripción del Estudio',
                'Modality': 'Modalidad',
                'InstitutionName': 'Institución',
                'ReferringPhysicianName': 'Médico Referente',
                'SeriesDescription': 'Descripción de la Serie',
                'SeriesNumber': 'Número de Serie',
                'ImageComments': 'Comentarios',
                'InstanceNumber': 'Número de Instancia'
            }
            
            # Intentar obtener información del dataset DICOM
            for tag, label in relevant_fields.items():
                try:
                    if hasattr(dicom, tag):
                        value = str(getattr(dicom, tag)).strip()
                        if value:  # Solo mostrar si hay un valor
                            # Formatear fechas
                            if 'Date' in tag and len(value) == 8:
                                value = f"{value[6:8]}/{value[4:6]}/{value[0:4]}"
                            # Formatear horas
                            elif 'Time' in tag and len(value) >= 6:
                                value = f"{value[0:2]}:{value[2:4]}:{value[4:6]}"
                            metadata_text += f"<tr><td><b>{label}:</b></td><td>{value}</td></tr>"
                except:
                    continue
            
            # Información técnica de la imagen
            metadata_text += "<tr><td colspan='2'><h3>Información Técnica</h3></td></tr>"
            
            technical_fields = {
                'Rows': 'Filas',
                'Columns': 'Columnas',
                'SamplesPerPixel': 'Muestras por Píxel',
                'PhotometricInterpretation': 'Interpretación Fotométrica',
                'PixelSpacing': 'Espaciado de Píxeles (mm)',
                'SliceThickness': 'Grosor de Corte (mm)',
                'SpacingBetweenSlices': 'Espacio entre Cortes (mm)',
                'ImageOrientationPatient': 'Orientación de la Imagen',
                'ImagePositionPatient': 'Posición de la Imagen',
                'PixelRepresentation': 'Representación de Píxeles',
                'BitsAllocated': 'Bits Asignados',
                'BitsStored': 'Bits Almacenados',
                'HighBit': 'Bit Alto',
                'WindowCenter': 'Centro de Ventana',
                'WindowWidth': 'Ancho de Ventana',
                'RescaleIntercept': 'Intercepto de Reescalado',
                'RescaleSlope': 'Pendiente de Reescalado'
            }
            
            for tag, label in technical_fields.items():
                try:
                    if hasattr(dicom, tag):
                        value = str(getattr(dicom, tag)).strip()
                        if value:  # Solo mostrar si hay un valor
                            metadata_text += f"<tr><td><b>{label}:</b></td><td>{value}</td></tr>"
                except:
                    continue
            
            # Agregar dimensiones de la imagen actual
            if hasattr(dicom, 'pixel_array'):
                shape = dicom.pixel_array.shape
                if len(shape) == 2:
                    metadata_text += f"<tr><td><b>Dimensiones:</b></td><td>{shape[0]} x {shape[1]} píxeles</td></tr>"
                elif len(shape) == 3:
                    metadata_text += f"<tr><td><b>Dimensiones:</b></td><td>{shape[0]} x {shape[1]} x {shape[2]} píxeles</td></tr>"
            
            metadata_text += "</table>"
            
            # Actualizar el widget de metadata
            self.metadata_text.setText(metadata_text)
            
        except Exception as e:
            self.metadata_text.setText(f"Error al mostrar metadata: {str(e)}")

    def load_dicom_folder(self):
        """Cargar una carpeta de archivos DICOM"""
        try:
            folder_path = QFileDialog.getExistingDirectory(
                self, "Seleccionar Carpeta de Imágenes", "",
                QFileDialog.Option.ShowDirsOnly)
            
            if not folder_path:
                return
                
            # Obtener lista de archivos en la carpeta
            all_files = sorted(os.listdir(folder_path))
            dicom_files = []
            
            # Mostrar progreso de búsqueda
            progress = QProgressDialog("Buscando archivos DICOM...", "Cancelar", 0, len(all_files), self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            
            for i, file in enumerate(all_files):
                if progress.wasCanceled():
                    return
                    
                full_path = os.path.join(folder_path, file)
                if os.path.isfile(full_path):
                    try:
                        # Intentar leer como DICOM sin importar la extensión
                        dicom = pydicom.dcmread(full_path, force=True)
                        if hasattr(dicom, 'pixel_array'):
                            dicom_files.append((full_path, dicom))
                    except:
                        continue
                progress.setValue(i + 1)
            
            progress.close()
            
            if not dicom_files:
                QMessageBox.warning(self, "Error", 
                                  "No se encontraron archivos DICOM válidos en la carpeta")
                return
            
            # Mostrar progreso de carga
            progress = QProgressDialog("Cargando imágenes...", "Cancelar", 0, len(dicom_files), self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            
            # Usar el primer archivo para metadata
            self.display_metadata(dicom_files[0][1])
            
            # Procesar todas las imágenes
            processed_images = []
            for i, (file_path, dicom) in enumerate(dicom_files):
                if progress.wasCanceled():
                    break
                    
                try:
                    processed = self.process_dicom_image(dicom.pixel_array, dicom)
                    processed_images.append(processed)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
                    continue
                
                progress.setValue(i + 1)
            
            progress.close()
            
            if not processed_images:
                QMessageBox.warning(self, "Error", 
                                  "No se pudieron cargar las imágenes DICOM")
                return
            
            # Verificar que todas las imágenes tengan el mismo tamaño
            shapes = [img.shape for img in processed_images]
            if len(set(shapes)) > 1:
                QMessageBox.warning(self, "Error", 
                                  "Las imágenes tienen diferentes dimensiones")
                return
            
            # Convertir lista de imágenes a array 3D
            self.original_image = np.stack(processed_images)
            self.current_image = self.original_image.copy()
            
            # Configurar la navegación según dimensiones
            if len(self.original_image.shape) == 4:  # Para imágenes 4D
                self.total_slices = self.original_image.shape[0]
                self.current_slice = 0
            elif len(self.original_image.shape) == 3 and self.original_image.shape[-1] != 3:
                self.total_slices = self.original_image.shape[0]
                self.current_slice = 0
            else:
                self.total_slices = 1
                self.current_slice = 0
            
            # Forzar la habilitación de los controles
            QApplication.processEvents()  # Asegurar que la UI se actualice
            
            # Configurar el slider
            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(self.total_slices - 1)
            self.slice_slider.setValue(0)
            self.slice_label.setText(f"Corte: 1/{self.total_slices}")
            
            # Habilitar explícitamente todos los controles
            self.prev_slice_button.setEnabled(True)
            self.next_slice_button.setEnabled(True)
            self.play_button.setEnabled(True)
            self.speed_selector.setEnabled(True)
            self.slice_slider.setEnabled(True)
            self.filters_combo.setEnabled(True)
            self.reset_filters_button.setEnabled(True)
            self.predictive_model_button.setEnabled(True)
            
            # Forzar la actualización de la UI
            QApplication.processEvents()
            
            # Mostrar primera imagen
            self.display_image()
            self.statusBar.showMessage(f"Carpeta cargada: {self.total_slices} imágenes")
            
        except Exception as e:
            self.statusBar.showMessage(f"Error al cargar la carpeta: {str(e)}")
            QMessageBox.critical(self, "Error", 
                               f"No se pudo cargar la carpeta:\n{str(e)}")

    def pause_playback(self):
        """Pausar la reproducción"""
        self.is_playing = False
        self.play_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_MediaPlay))
        self.play_button.setToolTip("Reproducir")
        self.play_timer.stop()
        # Mantener los botones habilitados
        self.prev_slice_button.setEnabled(True)
        self.next_slice_button.setEnabled(True)
        self.play_button.setEnabled(True)
        self.speed_selector.setEnabled(True)
        self.slice_slider.setEnabled(True)
        self.filters_combo.setEnabled(True)
        self.reset_filters_button.setEnabled(True)

    def stop_playback(self):
        """Detener reproducción y volver al primer slice"""
        self.pause_playback()
        if self.current_directory_files:
            self.current_file_index = 0
            self.load_dicom_file(self.current_directory_files[0])
        else:
            self.current_slice = 0
            self.slice_slider.setValue(0)
            self.slice_label.setText(f"Corte: 1/{self.total_slices}")
            self.display_image()

    def resizeEvent(self, event):
        """Manejar el redimensionamiento de la ventana"""
        super().resizeEvent(event)
        # Actualizar la imagen cuando se redimensiona la ventana
        if hasattr(self, 'current_image') and self.current_image is not None:
            self.display_image()
        # Reposicionar el cuadro de información
        self.position_image_info_label()

    def perform_segmentation(self):
        """Realizar segmentación en la imagen actual"""
        try:
            if self.current_image is None:
                QMessageBox.warning(self, "Error", "No hay imagen cargada")
                return

            # Mostrar diálogo de progreso
            progress = QProgressDialog("Realizando segmentación...", "Cancelar", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setValue(10)

            # Guardar temporalmente la imagen actual
            temp_path = "temp_dicom.dcm"
            try:
                # Crear un dataset DICOM temporal
                ds = pydicom.Dataset()
                ds.PixelData = self.current_image.tobytes()
                ds.Rows = self.current_image.shape[0]
                ds.Columns = self.current_image.shape[1]
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.PixelRepresentation = 0
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.save_as(temp_path)

                progress.setValue(30)

                # Realizar segmentación
                mascara = self.procesar_y_segmentar(temp_path)
                progress.setValue(70)

                # Convertir máscara a imagen visible
                mascara_visualizable = (mascara * 255).astype(np.uint8)

                # Crear una superposición coloreada
                imagen_original = self.current_image.copy()
                if len(imagen_original.shape) == 3:
                    imagen_original = imagen_original[self.current_slice]
                
                # Crear imagen RGB
                imagen_rgb = cv2.cvtColor(imagen_original, cv2.COLOR_GRAY2RGB)
                mascara_rgb = cv2.applyColorMap(mascara_visualizable, cv2.COLORMAP_JET)
                
                # Superponer con transparencia
                imagen_final = cv2.addWeighted(imagen_rgb, 0.7, mascara_rgb, 0.3, 0)

                progress.setValue(90)

                # Mostrar resultado en una nueva ventana
                cv2.imshow("Segmentación", imagen_final)
                
                # Guardar resultados
                cv2.imwrite("segmentacion_resultado.png", imagen_final)
                np.save("mascara_segmentacion.npy", mascara)

                progress.setValue(100)
                self.statusBar.showMessage("Segmentación completada. Resultados guardados.")

            finally:
                # Limpiar archivo temporal
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en la segmentación:\n{str(e)}")
            self.statusBar.showMessage("Error en la segmentación")
            print(f"Error en segmentación: {str(e)}")

    def setup_navigation_controls(self, enable):
        """Configurar controles de navegación"""
        # Habilitar/deshabilitar botones
        self.prev_slice_button.setEnabled(enable)
        self.next_slice_button.setEnabled(enable)
        self.play_button.setEnabled(enable)
        self.speed_selector.setEnabled(enable)
        self.slice_slider.setEnabled(enable)
        self.filters_combo.setEnabled(enable)
        self.reset_filters_button.setEnabled(enable)
        self.predictive_model_button.setEnabled(enable)
        
        if enable:
            # Configurar el slider
            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(len(self.current_directory_files) - 1)
            self.slice_slider.setValue(self.current_file_index)
            self.slice_label.setText(f"Archivo: {self.current_file_index + 1}/{len(self.current_directory_files)}")
            
            # Conectar señales
            self.prev_slice_button.clicked.connect(self.previous_slice)
            self.next_slice_button.clicked.connect(self.next_slice)
            self.play_button.clicked.connect(self.toggle_play)
            self.speed_selector.currentTextChanged.connect(self.change_playback_speed)
            self.slice_slider.valueChanged.connect(self.slider_changed)
        else:
            # Desconectar señales
            try:
                self.prev_slice_button.clicked.disconnect()
                self.next_slice_button.clicked.disconnect()
                self.play_button.clicked.disconnect()
                self.speed_selector.currentTextChanged.disconnect()
                self.slice_slider.valueChanged.disconnect()
            except:
                pass

    def setup_selection_tools(self):
        """Configurar herramientas de selección"""
        # Agregar botones a la barra de herramientas
        self.rect_select_button = QPushButton("Selección Rectangular")
        self.rect_select_button.setCheckable(True)  # Hacer el botón toggleable
        self.rect_select_button.clicked.connect(self.toggle_rect_selection)
        self.toolbar.addWidget(self.rect_select_button)
        
        self.circle_select_button = QPushButton("Selección Circular")
        self.circle_select_button.setCheckable(True)  # Hacer el botón toggleable
        self.circle_select_button.clicked.connect(self.toggle_circle_selection)
        self.toolbar.addWidget(self.circle_select_button)
        
        self.save_selection_button = QPushButton("Guardar Selección")
        self.save_selection_button.clicked.connect(self.save_current_selection)
        self.toolbar.addWidget(self.save_selection_button)
        
        self.show_selections_button = QPushButton("Mostrar Selecciones")
        self.show_selections_button.clicked.connect(self.show_selections)
        self.toolbar.addWidget(self.show_selections_button)
        
        # Agregar botón de detección de calcio
        self.detect_calcium_button = QPushButton("Detectar Calcio")
        self.detect_calcium_button.clicked.connect(self.detect_calcium)
        self.toolbar.addWidget(self.detect_calcium_button)
        
        # Configurar eventos del mouse para zoom y pan
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseMoveEvent = self.mouse_move_event
        self.image_label.mouseReleaseEvent = self.mouse_release_event
        
        # Habilitar todos los botones
        self.rect_select_button.setEnabled(True)
        self.circle_select_button.setEnabled(True)
        self.save_selection_button.setEnabled(True)
        self.show_selections_button.setEnabled(True)
        self.detect_calcium_button.setEnabled(True)

    def toggle_rect_selection(self):
        """Alternar el modo de selección rectangular"""
        if self.rect_select_button.isChecked():
            # Desactivar selección circular
            self.circle_select_button.setChecked(False)
            self.selection_tool.selection_type = "rectangular"
            self.statusBar.showMessage("Modo de selección: Rectangular")
        else:
            # Desactivar ambos modos
            self.selection_tool.selection_type = None
            self.statusBar.showMessage("Modo de selección: Desactivado")
        
        self.clear_selections()
        self.display_image()

    def toggle_circle_selection(self):
        """Alternar el modo de selección circular"""
        if self.circle_select_button.isChecked():
            # Desactivar selección rectangular
            self.rect_select_button.setChecked(False)
            self.selection_tool.selection_type = "circular"
            self.statusBar.showMessage("Modo de selección: Circular")
        else:
            # Desactivar ambos modos
            self.selection_tool.selection_type = None
            self.statusBar.showMessage("Modo de selección: Desactivado")
        
        self.clear_selections()
        self.display_image()

    def set_selection_type(self, selection_type):
        """Establecer el tipo de selección actual"""
        if selection_type == "rectangular":
            self.rect_select_button.setChecked(True)
            self.circle_select_button.setChecked(False)
            self.selection_tool.selection_type = "rectangular"
            self.statusBar.showMessage("Modo de selección: Rectangular")
        elif selection_type == "circular":
            self.rect_select_button.setChecked(False)
            self.circle_select_button.setChecked(True)
            self.selection_tool.selection_type = "circular"
            self.statusBar.showMessage("Modo de selección: Circular")
        else:
            self.rect_select_button.setChecked(False)
            self.circle_select_button.setChecked(False)
            self.selection_tool.selection_type = None
            self.statusBar.showMessage("Modo de selección: Desactivado")
        
        self.clear_selections()
        self.display_image()

    def calculate_image_coordinates(self, pos):
        """Calcular coordenadas de imagen a partir de coordenadas de pantalla"""
        if self.current_image is None:
            return None
            
        label_size = self.image_label.size()
        image_size = self.current_image.shape[:2][::-1]  # (width, height)
        
        # Calcular el tamaño de la imagen escalada
        scaled_width = int(image_size[0] * self.zoom_factor)
        scaled_height = int(image_size[1] * self.zoom_factor)
        
        # Calcular el offset del centro
        center_x = (label_size.width() - scaled_width) // 2
        center_y = (label_size.height() - scaled_height) // 2
        
        # Ajustar las coordenadas del mouse según el zoom y el pan
        adjusted_x = (pos.x() - center_x - self.pan_offset.x()) / self.zoom_factor
        adjusted_y = (pos.y() - center_y - self.pan_offset.y()) / self.zoom_factor
        
        # Verificar que las coordenadas estén dentro de los límites de la imagen
        if 0 <= adjusted_x < image_size[0] and 0 <= adjusted_y < image_size[1]:
            return QPoint(int(adjusted_x), int(adjusted_y))
        return None

    def update_coordinates_display(self, pos):
        """Actualizar la visualización de información completa de la imagen en tiempo real"""
        if self.current_image is None:
            self.image_info_label.setText("Cargando imagen...")
            return
            
        # Si no hay metadata DICOM, mostrar información básica
        if self.current_dicom_metadata is None:
            info_text = f"""Zoom: {int(self.zoom_factor * 100)}%
Sin metadata DICOM"""
            self.image_info_label.setText(info_text)
            return
            
        # Calcular coordenadas de imagen
        image_coords = self.calculate_image_coordinates(pos)
        
        if image_coords:
            # Obtener valor del píxel
            try:
                if len(self.current_image.shape) == 3:
                    pixel_value = self.current_image[image_coords.y(), image_coords.x()]
                else:
                    pixel_value = self.current_image[image_coords.y(), image_coords.x()]
                
                # Formatear el valor del píxel
                if isinstance(pixel_value, (int, float)):
                    pixel_value = f"{pixel_value:.0f}"
            except:
                pixel_value = "N/A"
            
            # Calcular coordenadas del mundo real (mm)
            world_x = "N/A"
            world_y = "N/A"
            try:
                if hasattr(self.current_dicom_metadata, 'PixelSpacing'):
                    pixel_spacing = self.current_dicom_metadata.PixelSpacing
                    if len(pixel_spacing) >= 2:
                        world_x = f"{image_coords.x() * pixel_spacing[0]:.2f}"
                        world_y = f"{image_coords.y() * pixel_spacing[1]:.2f}"
                        # Agregar unidades
                        world_x += " mm"
                        world_y += " mm"
            except:
                pass
            
            # Obtener parámetros DICOM
            try:
                # Parámetros de exposición
                ma = getattr(self.current_dicom_metadata, 'XRayTubeCurrent', 'N/A')
                kv = getattr(self.current_dicom_metadata, 'KVP', 'N/A')
                
                # Parámetros de ventana
                wl = getattr(self.current_dicom_metadata, 'WindowCenter', 'N/A')
                ww = getattr(self.current_dicom_metadata, 'WindowWidth', 'N/A')
                
                # Si WL y WW son listas, tomar el primer valor
                if isinstance(wl, list) and len(wl) > 0:
                    wl = wl[0]
                if isinstance(ww, list) and len(ww) > 0:
                    ww = ww[0]
                
            except:
                ma = kv = wl = ww = "N/A"
            
            # Construir texto de información
            info_text = f"""15 mA: {ma}
120.00kV: {kv}
X: {world_x}
Y: {world_y}
X: {image_coords.x()} px
Y: {image_coords.y()} px
Value: {pixel_value}
Zoom: {int(self.zoom_factor * 100)}%
WL: {wl}
WW: {ww}"""
            
            self.image_info_label.setText(info_text)
        else:
            # Si está fuera de la imagen, mostrar información básica
            try:
                ma = getattr(self.current_dicom_metadata, 'XRayTubeCurrent', 'N/A')
                kv = getattr(self.current_dicom_metadata, 'KVP', 'N/A')
                wl = getattr(self.current_dicom_metadata, 'WindowCenter', 'N/A')
                ww = getattr(self.current_dicom_metadata, 'WindowWidth', 'N/A')
                
                if isinstance(wl, list) and len(wl) > 0:
                    wl = wl[0]
                if isinstance(ww, list) and len(ww) > 0:
                    ww = ww[0]
                
                info_text = f"""15 mA: {ma}
120.00kV: {kv}
Zoom: {int(self.zoom_factor * 100)}%
WL: {wl}
WW: {ww}"""
                
                self.image_info_label.setText(info_text)
            except:
                self.image_info_label.setText("Fuera de imagen")

    def clear_cache(self):
        """Limpiar el caché de imágenes"""
        self.image_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def is_selection_active(self):
        """Verificar si alguna herramienta de selección está activa"""
        return (hasattr(self, 'rect_select_button') and self.rect_select_button.isChecked()) or \
               (hasattr(self, 'circle_select_button') and self.circle_select_button.isChecked())

    def detect_calcium(self):
        """Detectar calcificaciones en la imagen actual"""
        if self.current_image is None:
            QMessageBox.warning(self, "Error", "No hay imagen cargada")
            return
            
        try:
            # Importar las funciones necesarias
            from dicom_viewer.processing import (
                detectar_calcificaciones,
                calcular_score_agatston
            )
            
            # Obtener la imagen actual
            if len(self.current_image.shape) == 3:
                current_slice = self.current_image[self.current_slice]
            else:
                current_slice = self.current_image
                
            # Convertir a formato adecuado para el procesamiento
            if current_slice.dtype != np.int16:
                current_slice = current_slice.astype(np.int16)
            
            # Crear un volumen 3D con la imagen actual
            volumen = np.stack([current_slice], axis=0)
            
            # Detectar calcificaciones
            mask = detectar_calcificaciones(volumen, umbral=139, min_size=5, sigma=0.7)
            
            # Calcular score de Agatston
            score = calcular_score_agatston(mask, volumen, pixel_mm=0.5)
            
            # Mostrar resultados
            QMessageBox.information(self, "Resultados de Detección", 
                                  f"Score de Agatston: {score:.2f}\n\n"
                                  "Las calcificaciones detectadas se han marcado en rojo en la imagen.")
            
            # Actualizar la imagen con las calcificaciones marcadas
            if len(self.current_image.shape) == 3:
                self.current_image[self.current_slice] = self.marcar_calcificaciones(
                    self.current_image[self.current_slice], mask[0])
            else:
                self.current_image = self.marcar_calcificaciones(
                    self.current_image, mask[0])
            
            self.display_image()
            
        except ImportError as e:
            self.statusBar.showMessage(f"Error al importar módulos necesarios: {str(e)}")
            QMessageBox.critical(self, "Error", 
                               f"Error al importar módulos necesarios:\n{str(e)}\n\n"
                               "Asegúrese de que el archivo ScoreCalcioTest3.py está en la carpeta processing.")
        except Exception as e:
            self.statusBar.showMessage(f"Error en detección de calcio: {str(e)}")
            QMessageBox.critical(self, "Error", 
                               f"Error durante la detección de calcio:\n{str(e)}")

    def marcar_calcificaciones(self, imagen, mascara):
        """Marcar las calcificaciones detectadas en la imagen"""
        # Convertir a RGB si es necesario
        if len(imagen.shape) == 2:
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
        else:
            imagen_rgb = imagen.copy()
        
        # Marcar las calcificaciones en rojo
        imagen_rgb[mascara] = [255, 0, 0]  # Rojo en RGB
        
        return imagen_rgb

def process_chunk(chunk_data):
    """Procesar un chunk de archivos DICOM"""
    chunk, directory = chunk_data
    valid_files = []
    for file_name in chunk:
        file_path = os.path.join(directory, file_name)
        try:
            # Verificación rápida del header
            with open(file_path, 'rb') as f:
                header = f.read(132)
                if b'DICM' in header or b'DICOM' in header or file_name.startswith('selection_'):
                    try:
                        dicom = pydicom.dcmread(file_path, force=True)
                        if hasattr(dicom, 'pixel_array') and dicom.pixel_array.size > 0:
                            valid_files.append(file_path)
                    except:
                        continue
        except:
            continue
    return valid_files
