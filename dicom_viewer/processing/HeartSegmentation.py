import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import backend as K
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
import os
import numpy as np
import cv2
import pydicom as dicom
import pandas as pd
from glob import glob
from tqdm import tqdm
if TF_AVAILABLE:
    from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
"""
The "IoU coefficient" refers to the Intersection over Union (IoU) metric, a common evaluation metric in computer vision for measuring the overlap between
two bounding boxes or segmented regions, like predicted and ground truth masks. It's calculated as the area of the intersection of the two regions divided
by their total union, resulting in a score between 0 (no overlap) and 1 (perfect overlap).
"""
def iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-15) / (union + 1e-15)

"""
The Dice coefficient, also known as the Sørensen-Dice coefficient, is a statistical measure used to compare the similarity between two sets of data,
ranging from 0 (no similarity) to 1 (perfect match). It is calculated using the formula 2 * |Intersection| / (|Set A| + |Set B|)
"""
smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    for r in ranges:
        new_subdir = os.path.join(path, r)
        if not os.path.exists(new_subdir):
            os.makedirs(new_subdir)

"""
Functions to convert DCM to BMP
"""
def _get_LUT_value_LINEAR_EXACT(data, window, level):
    data_min = data.min()
    data_max = data.max()
    data_range = data_max - data_min
    data = np.piecewise(data,
                        [data <= (level - (window) / 2),
                         data > (level + (window) / 2)],
                        [data_min, data_max,
                         lambda data: ((data - level + window / 2) / window * data_range) + data_min])
    return data

def _pixel_process(ds, pixel_array):
    if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
        rescale_slope = float(ds.RescaleSlope)  # int(ds.RescaleSlope)
        rescale_intercept = float(ds.RescaleIntercept)  # int(ds.RescaleIntercept)
        pixel_array = (pixel_array) * rescale_slope + rescale_intercept
    else:
        pixel_array = apply_modality_lut(pixel_array, ds)

    if 'VOILUTFunction' in ds and ds.VOILUTFunction == 'SIGMOID':
        pixel_array = apply_voi_lut(pixel_array, ds)
    elif 'WindowCenter' in ds and 'WindowWidth' in ds:
        window_center = ds.WindowCenter
        window_width = ds.WindowWidth

        if type(window_center) == dicom.multival.MultiValue:
            window_center = float(window_center[0])
        else:
            window_center = float(window_center)
        if type(window_width) == dicom.multival.MultiValue:
            window_width = float(window_width[0])
        else:
            window_width = float(window_width)
        pixel_array = _get_LUT_value_LINEAR_EXACT(pixel_array, window_width, window_center)
    else:
        pixel_array = apply_voi_lut(pixel_array, ds)

    pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())) * 255.0

    if 'PhotometricInterpretation' in ds and ds.PhotometricInterpretation == "MONOCHROME1":
        pixel_array = np.max(pixel_array) - pixel_array

    return pixel_array.astype('uint8')

def _is_unsupported(ds):
    try:
        if ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.104.1':
            return 'Encapsulated PDF Storage'
        elif ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.88.59':
            return 'Key Object Selection Document'
    except:
        pass
    return False

def _ds_to_file(file_path):
    ds = dicom.dcmread(file_path, force=True)

    is_unsupported = _is_unsupported(ds)
    if is_unsupported:
        rv = f'{file_path} cannot be converted.\n{is_unsupported} is currently not supported'
        return rv

    pixel_array = ds.pixel_array.astype(float)

    if len(pixel_array.shape) == 3 and pixel_array.shape[2] != 3:
        rv = f'{file_path} cannot be converted.\nMultiframe images are currently not supported'
        return rv

    pixel_array = _pixel_process(ds, pixel_array)

    if 'PhotometricInterpretation' in ds and ds.PhotometricInterpretation in ['YBR_RCT', 'RGB', 'YBR_ICT',
                                                                              'YBR_PARTIAL_420', 'YBR_FULL_422',
                                                                              'YBR_FULL', 'PALETTE COLOR']:
        pixel_array[:, :, [0, 2]] = pixel_array[:, :, [2, 0]]

    return pixel_array


def filter_heart(image, mask):
    MIN_AREA_PIXELES = 100

    if mask.dtype != 'uint8':
        mask = mask.astype('uint8')

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contornos_validos = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area > MIN_AREA_PIXELES:
            contornos_validos.append(c)

    if not contornos_validos:
        final_mask = np.zeros_like(mask)
        final_result = cv2.bitwise_and(image, image, mask=final_mask)
        final_result = np.expand_dims(final_result, axis=-1)
        return final_result

    puntos_unificados = np.vstack(contornos_validos)

    x, y, w, h = cv2.boundingRect(puntos_unificados)

    final_mask = np.zeros_like(mask)
    cv2.rectangle(final_mask, (x, y), (x + w, y + h), 255, -1) # -1 para for fill rectangle.

    final_result = cv2.bitwise_and(image, image, mask=final_mask)
    final_result = np.expand_dims(final_result, axis=-1)

    return final_result

def filter_noise(image):
    image = image.astype('uint8')

    umbral, img_binaria = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contornos, jerarquia = cv2.findContours(img_binaria, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        final_mask = np.zeros_like(image)
        final_result = cv2.bitwise_and(image, image, mask=final_mask)
        final_result = np.expand_dims(final_result, axis=-1)
        return final_result

    mayor_area = 0
    indice_mayor_contorno = -1
    for i, contorno in enumerate(contornos):
        if jerarquia[0][i][3] == -1:
            area = cv2.contourArea(contorno)
            if area > mayor_area:
                mayor_area = area
                indice_mayor_contorno = i
    if indice_mayor_contorno == -1:
        final_mask = np.zeros_like(image)
        final_result = cv2.bitwise_and(image, image, mask=final_mask)
        final_result = np.expand_dims(final_result, axis=-1)
        return final_result

    mascara = np.zeros_like(image, dtype=np.uint8)

    cv2.drawContours(mascara, contornos, indice_mayor_contorno, (255), cv2.FILLED)

    for i, contorno in enumerate(contornos):
        if jerarquia[0][i][3] == indice_mayor_contorno:
            cv2.drawContours(mascara, contornos, i, (255), cv2.FILLED)

    imagen_final = cv2.bitwise_and(image, image, mask=mascara)
    imagen_final = np.expand_dims(imagen_final, axis=-1)

    return imagen_final

def filter_calcium(imagen):
    UMBRAL_INTENSIDAD = 200
    ITERACIONES_DILATACION = 1
    AREA_MINIMA_PARA_DILATAR = 1000
    AREA_MAXIMA_PIXELES_FINAL = 2000

    imagen = imagen.astype('uint8')

    _, imagen_umbralizada = cv2.threshold(imagen, UMBRAL_INTENSIDAD, 255, cv2.THRESH_BINARY)

    imagen_dilatada_selectiva = np.zeros_like(imagen_umbralizada)

    contornos_iniciales, _ = cv2.findContours(imagen_umbralizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kernel = np.ones((5, 5), np.uint8)

    for contorno in contornos_iniciales:
        area_contorno = cv2.contourArea(contorno)

        if area_contorno > AREA_MINIMA_PARA_DILATAR:
            mascara_contorno = np.zeros_like(imagen_umbralizada)
            cv2.drawContours(mascara_contorno, [contorno], -1, 255, -1)

            region_para_dilatar = cv2.bitwise_and(imagen_umbralizada, imagen_umbralizada, mask=mascara_contorno)
            region_dilatada = cv2.dilate(region_para_dilatar, kernel, iterations=ITERACIONES_DILATACION)

            imagen_dilatada_selectiva = cv2.bitwise_or(imagen_dilatada_selectiva, region_dilatada)
        else:
            cv2.drawContours(imagen_dilatada_selectiva, [contorno], -1, 255, -1)

    contornos_finales, _ = cv2.findContours(imagen_dilatada_selectiva, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    imagen_final = imagen_dilatada_selectiva.copy()

    for contorno in contornos_finales:
        area_final = cv2.contourArea(contorno)

        if area_final > AREA_MAXIMA_PIXELES_FINAL:
            cv2.drawContours(imagen_final, [contorno], -1, (0, 0, 0), -1)

    imagen_final = np.expand_dims(imagen_final, axis=-1)

    return imagen_final

def predict():
    MAIN_DIR = "PATIENTS"
    SOURCE_DCM =  MAIN_DIR + "/" + "22"
    PREPROC_DCM = SOURCE_DCM + "_PREPROC"
    MODEL_RESULTS = SOURCE_DCM + "_HEARTSEGM_RESULTS"
    #ranges = {'NULO': [0, 0.01], 'MINIMO': [0.01, 0.1], 'MEDIO': [0.1, 0.2], 'MODERADO': [0.2, 0.3], 'SEVERO': [0.3, 5]}
    ranges = {'TEST': [0, 5]}

    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir(MODEL_RESULTS)
    create_dir(PREPROC_DCM)

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/model.h5")

    """ Load the dataset """
    test_x = glob(SOURCE_DCM+"/*.dcm")
    print(f"Test: {len(test_x)}")

    # IMAGES SUM
    img_total = np.zeros((512, 512, 1))
    factor = 1.0

    """ Loop over the data """
    for file_x in tqdm(test_x):
        """ Extract the names """
        name = os.path.basename(file_x).replace('.dcm', '')

        #area = float(name.split("_")[3])
        area = 0

        category = ''
        for r in ranges:
            if area >= ranges[r][0] and area < ranges[r][1]:
                category = r

        """ Read the image """
        image = dicom.dcmread(file_x).pixel_array
        image = np.expand_dims(image, axis=-1)
        image = image/np.max(image) * 255.0
        x = image/255.0
        x = np.concatenate([x, x, x], axis=-1)
        x = np.expand_dims(x, axis=0)

        """ Prediction """
        mask = model.predict(x)[0]
        mask = mask > 0.5
        mask = mask.astype(np.int32)
        mask = mask * 255

        bmp_image = _ds_to_file(file_x)
        bmp_image = np.expand_dims(bmp_image, axis=-1)

        final_image = filter_heart(bmp_image, mask) #ORIGINAL: image
        final_image = filter_noise(final_image)
        heart_image = final_image.copy()
        final_image = filter_calcium(heart_image)
        cat_images = np.concatenate([bmp_image, heart_image, final_image], axis=1) #ORIGINAL: image

        EXTENSION = ".bmp"
        cv2.imwrite(f"{MODEL_RESULTS}/{category}/{name}{EXTENSION}", cat_images)
        cv2.imwrite(f"{PREPROC_DCM}/{category}/{name}{EXTENSION}", final_image)

        #IMAGES SUM
        final_image = final_image.astype(float) * factor
        img_total = cv2.add(img_total, final_image)
        factor = factor - 0.05
        if factor < 0:
            factor = 0
        cv2.imwrite(PREPROC_DCM + "/IMAGE_TOTAL.bmp", img_total)
    
    """ Creating a directory """
def create_dir_results(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(image, mask, y_pred, save_image_path):
    ## i - m - y
    line = np.ones((H, 10, 3)) * 128

    """ Mask """
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)

    """ Predicted Mask """
    y_pred = np.expand_dims(y_pred, axis=-1)    ## (512, 512, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  ## (512, 512, 3)
    y_pred = y_pred * 255

    cat_images = np.concatenate([image, line, mask, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

def eval():
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/model.h5")

    """ Load the dataset """
    test_x = sorted(glob(os.path.join("new_data", "valid", "image", "*")))
    test_y = sorted(glob(os.path.join("new_data", "valid", "mask", "*")))
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Evaluation and Prediction """
    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x = image/255.0
        x = np.expand_dims(x, axis=0)

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        y = mask/255.0
        y = y > 0.5
        y = y.astype(np.int32)

        """ Prediction """
        y_pred = model.predict(x)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)

        """ Saving the prediction """
        save_image_path = f"results/{name}.png"
        save_results(image, mask, y_pred, save_image_path)

        """ Flatten the array """
        y = y.flatten()
        y_pred = y_pred.flatten()

        """ Calculating the metrics values """
        acc_value = accuracy_score(y, y_pred)
        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
        recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

    """ Metrics values """
    score = [s[1:]for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")

    df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv("files/score.csv")