# Deeplearning-based radiomics

from __future__ import print_function
import logging
import os
import pandas
import SimpleITK as sitk
import numpy as np
import collections
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import os,sys
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import Model

# CNN
base_model = ResNet50(weights='imagenet', include_top=True)
from tensorflow.keras.models import Model
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Feature Extraction
# Cropping box
def maskcroppingbox(images_array, use2D=False):
    images_array_2 = np.argwhere(images_array)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = images_array_2.min(axis=0), images_array_2.max(axis=0) + 1
    empty = []
    for i in range(zstart, zstop):
        test_array = images_array[i, :, :]
        num = test_array.sum()
        empty.append(num)
    max_index = empty.index(max(empty)) + 1
    return (zstart, ystart, xstart), (zstop, ystop, xstop), max_index


def featureextraction(imageFilepath, maskFilepath):
    image_array = sitk.GetArrayFromImage(imageFilepath)
    mask_array = sitk.GetArrayFromImage(maskFilepath)
    (zstart, ystart, xstart), (zstop, ystop, xstop), maxIndex = maskcroppingbox(mask_array, use2D=False)
    roi_images = image_array[zstart - 1:zstop + 1, ystart:ystop, xstart:xstop].transpose((2, 1, 0))
    roi_images1 = zoom(roi_images, zoom=[224 / roi_images.shape[0], 224 / roi_images.shape[1], 1], order=3)
    roi_images2 = np.array(roi_images1, dtype=np.float)
    x = image.img_to_array(roi_images2)

    x_order = np.asarray(x[:, :, maxIndex])
    x_befor = np.asarray(x[:, :, maxIndex - 1])
    x_later = np.asarray(x[:, :, maxIndex + 1])
    a = np.asarray(x_order)
    b = np.asarray(x_befor)
    c = np.asarray(x_later)
    mylist = [b, a, c]
    x = np.asarray(mylist)
    x = np.expand_dims(x, axis=0)

    # print('multi channel x shape:', x.shape)  # multi channel x shape: (3, 1, 224, 224)
    x = np.transpose(x, (0, 2, 3, 1))
    # print('x1.shape:', x.shape)  # x1.shape: (1, 224, 224, 43)
    x = preprocess_input(x)
    base_model_pool_features = model.predict(x)
    features = base_model_pool_features[0]
    deeplearningfeatures = collections.OrderedDict()
    for ind_, f_ in enumerate(features):
        deeplearningfeatures[str(ind_)] = f_
    return deeplearningfeatures

def main():
  outPath = 'D:/Radiomics-Feature-Extractor/Deep-learning-based-radiomics/example2/'

  inputCSV = os.path.join(outPath, 'data.csv')
  outputFilepath = os.path.join(outPath+'/'+'Resnet50_t1.csv')

  # Configure logging
  rLogger = logging.getLogger('DLradiomics')

  # Initialize logging for batch log messages
  logger = rLogger.getChild('batch')

  # logger.info('pyradiomics version: %s', radiomics.__version__)
  logger.info('Loading CSV')

  # ####### Up to this point, this script is equal to the 'regular' batchprocessing script ########

  try:
    # Use pandas to read and transpose ('.T') the input data
    # The transposition is needed so that each column represents one test case. This is easier for iteration over
    # the input cases
    flists = pandas.read_csv(inputCSV).T
  except Exception:
    logger.error('CSV READ FAILED', exc_info=True)
    exit(-1)

  logger.info('Loading Done')
  logger.info('Patients: %d', len(flists.columns))

  # Instantiate a pandas data frame to hold the results of all patients
  results = pandas.DataFrame()

  for entry in flists:  # Loop over all columns (i.e. the test cases)
    logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)",
                entry + 1,
                len(flists),
                flists[entry]['Image'],
                flists[entry]['Mask'])

    imageFilepath = flists[entry]['Image']
    maskFilepath = flists[entry]['Mask']
    label = flists[entry].get('Label', None)

    if str(label).isdigit():
      label = int(label)
    else:
      label = None

    if (imageFilepath is not None) and (maskFilepath is not None):
      featureVector = flists[entry]  # This is a pandas Series
      featureVector['Image'] = os.path.basename(imageFilepath)
      featureVector['Mask'] = os.path.basename(maskFilepath)

      try:
        # PyRadiomics returns the result as an ordered dictionary, which can be easily converted to a pandas Series
        # The keys in the dictionary will be used as the index (labels for the rows), with the values of the features
        # as the values in the rows.
        # color_channel = 0
        im=sitk.ReadImage(imageFilepath)
        mask=sitk.ReadImage(maskFilepath)

        result = pandas.Series(featureextraction(im, mask))
        featureVector = featureVector.append(result)

      except Exception:
        logger.error('FEATURE EXTRACTION FAILED:', exc_info=True)

      # To add the calculated features for this case to our data frame, the series must have a name (which will be the
      # name of the column.
      featureVector.name = entry
      # By specifying an 'outer' join, all calculated features are added to the data frame, including those not
      # calculated for previous cases. This also ensures we don't end up with an empty frame, as for the first patient
      # it is 'joined' with the empty data frame.
      results = results.join(featureVector, how='outer')  # If feature extraction failed, results will be all NaN

  logger.info('Extraction complete, writing CSV')
  # .T transposes the data frame, so that each line will represent one patient, with the extracted features as columns
  results.T.to_csv(outputFilepath, index=False, na_rep='NaN')
  logger.info('CSV writing complete')


if __name__ == '__main__':
  main()
