# Hand-crafted radiomics

import numpy as np
import collections
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import os,sys
import pandas as pd
import radiomics
from radiomics import featureextractor

## Set GPU
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Input data
imgDir = 'D:/Radiomics-Feature-Extractor/Hand-crafted-radiomics/data/'
dirlist = os.listdir(imgDir)[1:]
print(dirlist)

# read images in Nifti format 
def loadSegArraywithID(fold,iden):
    
    path = fold
    pathList = os.listdir(path)
    
    segPath = [os.path.join(path,i) for i in pathList if ('label' in i.lower()) & (iden in i.lower())][0]
    seg = sitk.ReadImage(segPath)
    return seg
# read regions of interest (ROI) in Nifti format 
def loadImgArraywithID(fold,iden):
    
    path = fold
    pathList = os.listdir(path)
    
    imgPath = [os.path.join(path,i) for i in pathList if ('t1' in i.lower()) & (iden in i.lower())][0]
    img = sitk.ReadImage(imgPath)    
    return img

# Feature Extraction
featureDict = {}
for ind in range(len(dirlist)):
    path = os.path.join(imgDir,dirlist[ind])
    
    # you can make your own pipeline to import data, but it must be SimpleITK images
    mask = loadSegArraywithID(path,'label')  # see line 26 !
    img = loadImgArraywithID(path,'t1')      # see line 35 !
    params = './tumor.yaml'
    
    extractor = featureextractor.RadiomicsFeatureExtractor(params)

    result = extractor.execute(img,mask)
    key = list(result.keys())
    key = key[1:] 
    
    feature = []
    for jind in range(len(key)):
        feature.append(result[key[jind]])
        
    featureDict[dirlist[ind]] = feature
    dictkey = key
    print(dirlist[ind])

# Output  
dataframe = pd.DataFrame.from_dict(featureDict, orient='index', columns=dictkey)
dataframe.to_csv('D:/Radiomics-Feature-Extractor/Hand-crafted-radiomics/method-3/feature.csv')