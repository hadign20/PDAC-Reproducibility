import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import SimpleITK as sitk
from scipy.ndimage import zoom

def load_model(model_name):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained = True)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained = True)
    else:
        raise ValueError("Unsupported model name.")

    model = nn.Sequential(*list(model.children())[:-1])  # Remove the classification layer
    model.eval()
    return model


def preprocess_image(img_array, mask_array):
    img_array[mask_array == 0] = 0

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img_array)
    img_tensor = img_tensor.unsqueeze(0) # Add batch dimension
    return img_tensor

def load_and_preprocess_image(image_path, mask_path):
    img = sitk.ReadImage(image_path)
    img_array = sitk.GetArrayFromImage(img)
    img_array = zoom(img_array, (224/img_array.shape[0], 224/img_array.shape[1], 1), order=3)

    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)
    mask_array = zoom(mask_array, (224 / mask_array.shape[0], 224 / mask_array.shape[1], 1), order=0)

    img_tensor = preprocess_image(img_array, mask_array)
    return img_tensor


def extract_deep_features(image_path, mask_path, model_name='resnet50'):
    model = load_model(model_name)
    img_tensor = load_and_preprocess_image(image_path, mask_path)
    with torch.no_grad():
        features = model(img_tensor).squeeze().numpy()
    return features

def save_deep_features(features, output_file):
    df = pd.DataFrame(features)
    df.to_csv(output_file, index=False)

def process_and_save_deep_features(image_dir, mask_dir, model_name, output_file):
    features_list = []
    for img_file in os.listdir(image_dir):
        if img_file.endswith('nii') or img_file.endswith('nii.gz') or img_file.endswith('mha'):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file)
            features = extract_deep_features(img_path, mask_path, model_name)
            features_list.append(features)
    save_deep_features(features_list, output_file)
