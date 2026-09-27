#from pathlib import Path
#import pickle
import numpy as np

def standardize_hu(volume, min_hu=-1000, max_hu=400, minmax_norm=False):
    """
    Clip the CT volume to a fixed HU range and optionally scale it to [0, 1].    

    -1000 is approximately the HU value of air, and 400 is a convention
    used to prevent high values from bone or other dense structures from
    compressing the values of interest when scaling to [0,1].
    There's a page on wikipedia with different values of HU for different things:
    https://en.wikipedia.org/wiki/Hounsfield_scale

    Both values are a commonly used convention in the literature, for example:
    https://www.mdpi.com/2076-3417/12/4/1959

    Normalization can speed of gradient descent convergence during training:
    https://pmc.ncbi.nlm.nih.gov/articles/PMC10361226/#sec2.2.4.3.6
    """
    #Clip HU values
    volume = np.clip(volume, min_hu, max_hu)

    # Min-max normalization
    if minmax_norm:
        volume=(volume-min_hu) / (max_hu-min_hu)

    return volume

#needs to me implemented later for the nns, for now pyradiomics offers an option to resample
def resample(image, mask):
    """
    Different acquisition protocols can produce different radiomics features,
    even when the same lung is scanned. One cause is the difference in pixel
    and slice spacing between CT scans.

    Resampling uses interpolation to estimate pixel or voxel values between
    known data points. This allows CT images to be resized to a
    selected voxel size, such as 1x1x1 mm, making radiomics features more
    comparable and reproducible across scans.

   (notas:
    - Imagem CT: interpolação spline, porque os valores HU são contínuos. linear e cubico?
    - Máscara: nearest-neighbour, para continuar binária (0 ou 1) e não criar valores intermédios.
    )
    References:
    https://pmc.ncbi.nlm.nih.gov/articles/PMC11835989/pdf/330_2024_Article_11093.pdf
    https://pmc.ncbi.nlm.nih.gov/articles/PMC10361226/ #sec2.2.4.3.2
    """
    #TODO

#needs to be implemented later for the nns(probably)
def resize_image():
    """
    Resizing is necessary when working with deep learning models that 
    require input images to comply with this pre-defined size:
    https://pmc.ncbi.nlm.nih.gov/articles/PMC10361226/#sec2.2.4.3.5 
    """
    #TODO

