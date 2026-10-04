#Note: The code has to be edited for extracting features from the entire dataset, this shouldn't be very hard but could require changing the arguments of the function
#Note: Files paths have to be adjusted before use
#      Those are on lines 25, 33, 125

#Note: I used a radius of 6mm instead of the one written in the dataset because the test files do not have that,
#      6mm is around the 97th percentile, the 95th percentile is around 5mm

#I used the following command to install featureextractor from radiomics
#pip install git+https://github.com/AIM-Harvard/pyradiomics.git

import numpy as np
import pandas as pd
import SimpleITK as sitk
import logging
from radiomics import featureextractor

# Configure pyradiomics logging to avoid redundant warnings in output
import radiomics
radiomics.setVerbosity(logging.ERROR)

def extract_features_for_series(lndb_id):
    print(f"\nProcessing Series LNDb-{lndb_id:04d}...")

    # 1. Load the SimpleITK image
    image_path = f'data0/LNDb-{lndb_id:04d}.mhd'
    try:
        image = sitk.ReadImage(image_path)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None

    # 2. Retrieve nodule coordinates from allNods.csv
    nods_df = pd.read_csv('allNods.csv')
    filtered_nods = nods_df[nods_df['LNDbID'] == lndb_id].copy()
    filtered_nods = filtered_nods.dropna(subset=['x', 'y', 'z'])

    if filtered_nods.empty:
        print(f"No valid nodule annotations found for LNDb-{lndb_id:04d}.")
        return None

    # Get image geometry parameters
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    image_array = sitk.GetArrayFromImage(image) # depth, height, width

    rows = []
    # Iterate over all valid nodules for this series
    for idx, nodule in filtered_nods.iterrows():
        row = {
            "patient_id": "LNDb-{lndb_id:04d}",
            "nodule_id": idx,
        }
        finding_id = int(nodule['FindingID']) if 'FindingID' in nodule and not pd.isna(nodule['FindingID']) else idx
        malignancy_score = nodule['Malignancy'] if 'Malignancy' in nodule and not pd.isna(nodule['Malignancy']) else 'NaN'

        print(f"  - Extracting features for Nodule ID: {finding_id}, Malignancy: {malignancy_score}...")
        center_physical = np.array([nodule['x'], nodule['y'], nodule['z']])
        radius_mm = 6.0 #95 percentile is around 5mm, 6mm is around 97% percentile

        # Calculate pixel coordinates for the nodule center to find the central slice
        pixel_z_center = (center_physical[2] - origin[2]) / spacing[2]
        nodule_slice_idx = int(round(pixel_z_center))

        # Ensure the slice index is within bounds
        nodule_slice_idx = max(0, min(nodule_slice_idx, image_array.shape[0] - 1))

        # 3. Create a binary 2D mask on the central nodule slice
        # Initialize a 3D mask array with zeros
        mask_array = np.zeros(image_array.shape, dtype=np.uint8)

        # Create 2D meshgrid for the selected slice (y, x dimensions)
        y_coords_2d, x_coords_2d = np.indices(image_array.shape[1:])

        # Convert 2D pixel coordinates of the slice to physical coordinates
        # Only use the y and x components of the origin and spacing
        physical_y_2d = origin[1] + y_coords_2d * spacing[1]
        physical_x_2d = origin[0] + x_coords_2d * spacing[0]

        # Calculate squared distance to the nodule center in 2D (for the specific slice)
        dist_squared_2d = ((physical_x_2d - center_physical[0]) ** 2 +
                           (physical_y_2d - center_physical[1]) ** 2)

        # Mark voxels inside the nodule radius circle as 1 on the selected slice
        mask_array[nodule_slice_idx, :, :][dist_squared_2d <= (radius_mm ** 2)] = 1        # If the generated mask is completely empty, skip it
        if np.sum(mask_array) == 0:
            print(f"    Warning: Generated mask is empty for nodule {finding_id}. Skipping.")
            continue

        mask_image = sitk.GetImageFromArray(mask_array)
        mask_image.CopyInformation(image)

        # 4. Initialize Radiomics Extractor & Extract features
        extractor = featureextractor.RadiomicsFeatureExtractor("causey_2d_vibes.yaml")
        extractor.enableAllFeatures()

        # Enable specific filters to generate ~1000 features in total
        extractor.enableImageTypeByName('Original')
        extractor.enableImageTypeByName('Wavelet')
        extractor.enableImageTypeByName('Square')
        extractor.enableImageTypeByName('SquareRoot')
        extractor.enableImageTypeByName('Logarithm')
        extractor.enableImageTypeByName('Exponential')
        extractor.enableImageTypeByName('Gradient')

        try:
            result = extractor.execute(image, mask_image)
            # Convert results to DataFrame and insert identifiers
            row.update(result)
            row["malignancy"] = malignancy_score
        except Exception as e:
            print(f"    Error during extraction for nodule {finding_id}: {e}")
        rows.append(row)

    return rows


def main():
    all_features = []
    for lndb_id in [1]:
        df = extract_features_for_series(lndb_id)
        if df is not None:
            all_features.extend(df)
            
    if all_features:
        csv_path = 'extracted_features.csv'
        df = pd.DataFrame(all_features)
        df.to_csv(csv_path, index=False)

        print(f"\nSuccess! Features from files saved to: {csv_path}")
        print(f"Shape of the dataset (Rows, Columns): {df.shape}")
        print(df.head(15))

if __name__ == "__main__":
    main()
