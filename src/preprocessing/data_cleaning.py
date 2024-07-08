import pandas as pd
from sklearn.impute import SimpleImputer
import torchvision.transforms as transforms
import SimpleITK as sitk

def clean_clinical_data(df):
    imputer = SimpleImputer(strategy='mean')
    df_cleaned = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_cleaned

def normalize_radiomics_features(df):
    return (df - df.mean()) / df.std()


def resample_image(image, reference_image):
    """Function to resample images to the same shape"""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    return resampler.Execute(image)

