# Deeplearning-based radiomics

import numpy as np
import collections
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import os,sys
import pandas as pd
from keras.preprocessing import image
 
## Set GPU
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Load model
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import Model

#fc
base_model = ResNet50(weights='imagenet', include_top=True)
from keras.models import Model
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)


# Load batch file

imgDir = './data/'
dirlist = os.listdir(imgDir)



# read images in Nifti format 
def loadSegArraywithID(fold,iden):
    
    path = fold
    pathList = os.listdir(path)
    segPath = [os.path.join(path,i) for i in pathList if ('seg' in i.lower()) & (iden in i.lower())][0]
    seg = sitk.ReadImage(segPath)
    return seg
# read regions of interest (ROI) in Nifti format 
def loadImgArraywithID(fold,iden):
    
    path = fold
    pathList = os.listdir(path)
    
    imgPath = [os.path.join(path,i) for i in pathList if ('im' in i.lower()) & (iden in i.lower())][0]
    img = sitk.ReadImage(imgPath)
    return img

# Feature Extraction
#Cropping box
def maskcroppingbox(images_array, use2D=False):
    images_array_2 = np.argwhere(images_array)
    print(images_array_2.shape)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = images_array_2.min(axis=0), images_array_2.max(axis=0) + 1
    print(zstart, ystart, xstart)
    print(zstop, ystop, xstop)
    return (zstart, ystart, xstart), (zstop, ystop, xstop)


def featureextraction(imageFilepath,maskFilepath):
    image_array = sitk.GetArrayFromImage(imageFilepath)
    mask_array = sitk.GetArrayFromImage(maskFilepath)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = maskcroppingbox(mask_array, use2D=False)
    roi_images = image_array[zstart-1:zstop+1,ystart:ystop,xstart:xstop].transpose((2,1,0))
    roi_images1 = zoom(roi_images, zoom=[224/roi_images.shape[0], 224/roi_images.shape[1],1], order=3)
    roi_images2 = np.array(roi_images1,dtype=np.float)
    x = image.img_to_array(roi_images2)
    num = []
    for i in range(zstart,zstop):
        mask_array = np.array(mask_array, dtype='uint8')
        images_array_3 = mask_array[:,:,i]
        num1 = images_array_3.sum()
        num.append(num1)
    maxindex = num.index(max(num))
    print(max(num),([*range(zstart,zstop)][num.index(max(num))]))
    x1 = np.asarray(x[:,:,maxindex-1])
    x2 = np.asarray(x[:,:,maxindex])#??????slice
    x3 = np.asarray(x[:,:,maxindex+1])
    print(x1.shape)
    x1 = np.expand_dims(x1, axis=0)
    x2 = np.expand_dims(x2, axis=0)
    x3 = np.expand_dims(x3, axis=0)
    a1 = np.asarray(x1)
    a2 = np.asarray(x2)
    a3 = np.asarray(x3)
    print(a1.shape)
    mylist = [a1,a2,a3]#????
    x = np.asarray(mylist)
    print(x.shape)
    x = np.transpose(x,(1,2,3,0))
    print(x.shape)
    x = preprocess_input(x)

    base_model_pool_features = model.predict(x)

    features = base_model_pool_features[0]

    deeplearningfeatures = collections.OrderedDict()
    for ind_,f_ in enumerate(features):
    	deeplearningfeatures[str(ind_)] = f_
    return deeplearningfeatures
 
featureDict = {}
for ind in range(len(dirlist)):
    path = os.path.join(imgDir,dirlist[ind])
    seg = loadSegArraywithID(path,'seg')
    im = loadImgArraywithID(path,'im')
        
    deeplearningfeatures = featureextraction(im,seg) 

    result = deeplearningfeatures
    key = list(result.keys())
    print(key)
    key = key[0:]
        
    feature = []
    for jind in range(len(key)):
        feature.append(result[key[jind]])
        
    featureDict[dirlist[ind]] = feature
    dictkey = key
    print(dirlist[ind])

    
# dataframe = pd.DataFrame.from_dict(featureDict, orient='index', columns='dictkey')
dataframe = pd.DataFrame.from_dict(featureDict, orient='index')
dataframe.to_csv('./feature.csv')



