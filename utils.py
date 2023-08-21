import os
import pandas as pd
import numpy as np
from skimage.io import imread

import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
  def __init__(self, name, csv_file, root_dir, transforms=None):
    self.name = name
    self.annotations = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.transforms = transforms
    self.categorical_features = ['anatom_site_general', 'benign_malignant', 'diagnosis', 'sex']
    self.column_indecies, self.column_values = self.get_column_information()
  
  def __len__(self):
    return(len(self.annotations))
  
  def __getitem__(self, index):
    """
    This function takes in the specified index of the dataset and returns 
    an image tensor and labels associated with image id as found in the CSV file.
    """
    image_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1] + '.JPG') # sets image path based on id
    image = imread(image_path)
    diagnosis_label = torch.from_numpy(self.get_onehot(self.annotations.iloc[index] , 'diagnosis').values.astype(np.float32))
    metadata = torch.cat(
              (torch.Tensor([self.annotations['age_approx'].iloc[index]]),
              torch.from_numpy(self.get_onehot(self.annotations.iloc[index], 'anatom_site_general').values.astype(np.float32)).flatten(),
              torch.from_numpy(self.get_onehot(self.annotations.iloc[index], 'sex').values.astype(np.float32)).flatten()),
              dim=0
    )

    metadata = torch.cat((metadata, metadata, metadata, metadata, metadata, metadata,
     metadata, metadata, metadata, metadata, metadata, metadata, metadata, 
     metadata, metadata, metadata, metadata, metadata, metadata, metadata,
     metadata, metadata, metadata, metadata, metadata, metadata, metadata,
     metadata, metadata, metadata, metadata, metadata, metadata, metadata), 0)

    if self.transforms:
      image = self.transforms(image)
    
    return (image, diagnosis_label, metadata)
  
  def get_column_information(self):
    """
    This function gets the dictionaries for the column indecies and values in order to be able
     to retrieve the one hot encoding of a particular attribute.
    """

    column_indecies = {}  # Mapping of feature -> start index of feature in a record
    column_values = {} # Mapping of feature -> list of categorical values the feature can take

    # build up the cat_index and cat_values dictionary
    for i, header in enumerate(self.annotations.keys()):
      for feature_name in self.categorical_features:
        if feature_name in header: # categorical header
            feature = feature_name
            value = header[len(feature) + 1:]
            if feature not in column_indecies:
                column_indecies[feature] = i
                column_values[feature] = [value]
            else:
                column_values[feature].append(value)
    return column_indecies, column_values

  def get_onehot(self, record, feature):
    """
    Return the portion of `record` that is the one-hot encoding
    of `feature`. For example, since the feature "work" is stored
    in the indices [5:12] in each record, calling `get_range(record, "work")`
    is equivalent to accessing `record[5:12]`.

    Args:
        - record: a numpy array representing one record, formatted
                  the same way as a row in `data.np`
        - feature: a string, should be an element of `catcols`
    """
    start_index = self.column_indecies[feature]
    stop_index = self.column_indecies[feature] + len(self.column_values[feature])
    return record[start_index:stop_index]
  
  def get_categorical_value(self, onehot, feature):
    """
    Return the categorical value name of a feature given
    a one-hot vector representing the feature.

    Args:
        - onehot: a numpy array one-hot representation of the feature
        - feature: a string, should be an element of `catcols`
    """
    
    categories = self.column_values[feature]
    return categories[torch.Tensor.int(onehot.argmax())]