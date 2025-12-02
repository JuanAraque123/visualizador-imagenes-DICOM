'''
Ejecucion:
python ScoreCalcioTest3.py "C:/FRAN_ROMERO/UELBOSQUE/Investigacion/SCORE_CALCIO/DATASET/DataSetCobos/SEGMENTS"
'''
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from skimage import measure, morphology, filters
from PIL import Image
import glob


def cargar_serie_dicom(carpeta):
    archivos = [pydicom.dcmread(os.path.join(carpeta, f), force=True)
                for f in os.listdir(carpeta) if f.lower().endswith('.dcm')]
    archivos.sort(key=lambda x: int(getattr(x, 'InstanceNumber', 0)))
    imgs = []
    for ds in archivos:
        img = ds.pixel_array.astype(np.int16)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        slope = getattr(ds, 'RescaleSlope', 1)
        #img = img * slope + intercept
        imgs.append(img)
    volumen = np.stack(imgs, axis=0)
    return volumen


def cargar_imagenes(carpeta):
    rutas = sorted(glob.glob(os.path.join(carpeta, '*.png')) +
                    glob.glob(os.path.join(carpeta, '*.jpg')) +
                    glob.glob(os.path.join(carpeta, '*.jpeg')))
    imgs = []
    for ruta in rutas:
        with Image.open(ruta) as img:
            gray = img.convert('L')
            arr = np.array(gray, dtype=np.int16)
            imgs.append(arr)
    if not imgs:
        raise ValueError(f"No se encontraron imágenes PNG/JPG en {carpeta}")
    volumen = np.stack(imgs, axis=0)
    return volumen


def cargar_volumen(carpeta):
    dcm_files = [f for f in os.listdir(carpeta) if f.lower().endswith('.dcm')]
    if dcm_files:
        return cargar_serie_dicom(carpeta)
    else:
        return cargar_imagenes(carpeta)


def detectar_calcificaciones(volumen, umbral=None, min_size=10, sigma=1.0):
    """
    Detecta regiones de calcificación por slice:
      - Aplica suavizado gaussiano (sigma).
      - Umbral adaptativo si umbral=None (Otsu), o fijo si se pasa.
      - Morfología: cierre, eliminación de objetos pequeños y llenado de agujeros.
    """
    masks = np.zeros_like(volumen, dtype=bool)
    for i in range(volumen.shape[0]):
        slice_img = volumen[i].astype(np.float32)
        # Suavizado para reducir ruido
        slice_blur = filters.gaussian(slice_img, sigma=sigma)
        # Umbral
        if umbral is None:
            thr = filters.threshold_otsu(slice_blur)
        else:
            thr = umbral
        mask = slice_blur > thr
        # Morfología: cerrar y filtrar
        #mask = morphology.binary_closing(mask, morphology.disk(2))
        mask = morphology.remove_small_objects(mask, min_size=min_size)
        mask = morphology.remove_small_holes(mask, area_threshold=min_size)
        masks[i] = mask
    return masks


def calcular_score_agatston(mask, volumen, pixel_mm=0.5):
    """
    Score de Agatston: suma de áreas ponderadas por intensidad máxima.
    """
    score = 0.0
    pixel_area = pixel_mm * pixel_mm
    for i in range(mask.shape[0]):
        lbl = measure.label(mask[i])
        props = measure.regionprops(lbl, intensity_image=volumen[i])
        for prop in props:
            area = prop.area * pixel_area
            max_int = prop.max_intensity
            if max_int < 200:
                w = 1
            elif max_int < 300:
                w = 2
            elif max_int < 400:
                w = 3
            else:
                w = 4
            score += area * w
    return score


def mostrar_resultados(volumen, mask, slice_idx=None):
    """
    Muestra una rebanada con superposición de contornos de calcificaciones.
    """
    if slice_idx is None:
        slice_idx = volumen.shape[0] // 2
    img = volumen[slice_idx]
    m = mask[slice_idx]
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')
    plt.contour(m, colors='r', linewidths=1.0)
    plt.title(f'Slice {slice_idx}: calcificaciones detectadas')
    plt.axis('off')
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Detección de calcificaciones coronarias en CT o imágenes')
    parser.add_argument('input_dir', help='Carpeta con DICOM o imágenes PNG/JPG')
    parser.add_argument('--slice', type=int, default=None, help='Slice a visualizar')
    parser.add_argument('--thr', type=float, default=139, help='Umbral fijo (HU o gris); si None usa Otsu por slice')
    parser.add_argument('--min_size', type=int, default=5, help='Área mínima de calcificación (pixeles)')
    parser.add_argument('--sigma', type=float, default=0.7, help='Sigma para suavizado gaussiano')
    parser.add_argument('--pixmm', type=float, default=0.5, help='Tamaño del píxel en mm para Agatston')
    args = parser.parse_args()

    print('[*] Cargando volumen...')
    vol = cargar_volumen(args.input_dir)
    print('[*] Detectando calcificaciones...')
    m = detectar_calcificaciones(vol, umbral=args.thr,
                                 min_size=args.min_size,
                                 sigma=args.sigma)
    print('[*] Calculando Score de Agatston...')
    score = calcular_score_agatston(m, vol, pixel_mm=args.pixmm)
    print(f'>> Score de Agatston: {score:.2f}')
    print('[*] Mostrando resultados...')
    mostrar_resultados(vol, m, slice_idx=args.slice)

if __name__ == '__main__':
    main()
