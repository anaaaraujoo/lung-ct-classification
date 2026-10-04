#Note: I only tested this on one patient, adjust the filters on lines 25-28 to change the patients
#Note: Please verify if the files paths on those lines are correct: 30,77
#Note: The file /root/.pylidcr has to contain the path to the dataset
#Note: Lines 17-19 are only here because I had some strange issues, tell me if you can run the program without them

#Question: Are citations necessary for the decision of the features we extract?
#Reminder: we still need to think about how we're selecting thes ROI, it probably needs to be consistent between datasets

import pylidc as pl
import SimpleITK as sitk
import numpy as np
from pylidc.utils import consensus
from radiomics import featureextractor
import pandas as pd
import os

import configparser
np.int = int
configparser.SafeConfigParser = configparser.ConfigParser

def main():
#----------Preprocessing/Feature Engineering----------------
# the pyradiomics pipeline is responsible for the resampling

    #scans = pl.query(pl.Scan)

    pid = 'LIDC-IDRI-0879'
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid)

    extractor = featureextractor.RadiomicsFeatureExtractor("/config/causey_2d_vibes.yaml")
    extractor.enableAllFeatures()
    extractor.enableImageTypeByName('Original')
    extractor.enableImageTypeByName('Wavelet')
    extractor.enableImageTypeByName('Square')
    extractor.enableImageTypeByName('SquareRoot')
    extractor.enableImageTypeByName('Logarithm')
    extractor.enableImageTypeByName('Exponential')
    extractor.enableImageTypeByName('Gradient')
    rows = []
    for scan in scans:
        nods = scan.cluster_annotations()
        vol = scan.to_volume(verbose=False)


        for nod_id, anns in enumerate(nods, start=1):
            row = {
                "patient_id": scan.patient_id,
                "nodule_id": nod_id,
            }
            # Perform a consensus consolidation and 50% agreement level.
            cmask,cbbox,_ = consensus(anns, clevel=0.5)

            # get the largest mask slice
            mask_slice_index = np.argmax(cmask.sum(axis=(0, 1)))
            scan_slice_index = cbbox[2].start + mask_slice_index

            # in the yaml file, the resampling happens on a 3d array
            slice_vol = vol[:,:,scan_slice_index:scan_slice_index+1]
            image = sitk.GetImageFromArray(np.transpose(slice_vol, (2,0,1)))
            image.SetSpacing((scan.pixel_spacing,scan.pixel_spacing,scan.slice_spacing))

            mask_array = np.zeros(slice_vol.shape, dtype=np.uint8)
            mask_array[cbbox[0], cbbox[1], 0] = cmask[:, :, mask_slice_index]
            mask_image = sitk.GetImageFromArray(np.transpose(mask_array, (2,0,1)))
            mask_image.CopyInformation(image)

            result = extractor.execute(image, mask_image)

            row.update(result)
            # we will experiment with both the mean and median
            ratings = [ann.malignancy for ann in anns]
            row["malignancy_mean"] = float(np.mean(ratings))
            row["malignancy_median"] = float(np.median(ratings))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("/extracted_features.csv", index=False)


if __name__=="__main__":
    main()
