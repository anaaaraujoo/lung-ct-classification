#just a quick baseline pipeline, this should be organized later, im not so sure about the distinction
# between data cleaning and the preprocessing i was supposed to do... In which steps they appear...
# for now i will leave it here like this:
import pylidc as pl
import SimpleITK as sitk
import numpy as np
from pylidc.utils import consensus
from radiomics import featureextractor
import pandas as pd


def main():
#----------Preprocessing/Feature Engineering----------------
# the pyradiomics pipeline is responsible for the resampling
    scans = pl.query(pl.Scan).limit(5)

    extractor = featureextractor.RadiomicsFeatureExtractor("configs/causey_2d_vibes.yaml")
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
    df.to_csv("data/radiomics_causey_2d.csv", index=False)
    

if __name__=="__main__":
    main()