import os
import openslide
#import pandas as pd

from clam.create_patches import seg_and_patch
from clam.datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP

'''
___Description___
Custom wrapper class for CLAM: Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images
Main code adapted from the original GitHub repository with minor changes: https://github.com/mahmoodlab/CLAM


___Usage___ 
Given a set of slides in a directory, we want to patch them on the fly by first extracting their coordinates.
The coordinates are then saved and called by OpenSlide to create the dataloader with the augmented image tiles. 
If there are labels available, we get them from a CSV file.


___Example___ 
from torchvision import transforms

slide_dir = './slide/'
coord_dir = './coord/'
save_dir = './save/'

patch_size = 256
step_size = 256
seg_params = {'seg_level': 1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': 'none', 'exclude_ids': 'none'}
filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}

Slide2Coord(slide_dir, coord_dir, save_dir, patch_size, step_size, seg_params, filter_params).get_coordinates()
dataset = Coord2Data(slide_dir, coord_dir, transforms=transforms.ToTensor(), csv_path=None).get_dataset()
'''

class Slide2Coord:

    def __init__(
        self,
        slide_dir,
        coord_dir,
        save_dir,
        patch_size,
        step_size,
        seg_params,
        filter_params,
        patch_level,
        ):

        self.slide_dir = slide_dir
        self.coord_dir = coord_dir
        self.save_dir = save_dir
        self.patch_size = patch_size
        self.step_size = step_size
        self.seg_params = seg_params
        self.filter_params = filter_params
        self.patch_level = patch_level

    def get_coordinates(self):
        """extracts and saves coordinates
           for all slides in the directory
        """

        seg_and_patch(
            source=self.slide_dir,
            save_dir=self.save_dir,
            patch_save_dir=self.coord_dir,
            mask_save_dir='',
            stitch_save_dir='',
            patch_size=self.patch_size,
            step_size=self.step_size,
            seg_params=self.seg_params,
            filter_params=self.filter_params,
            vis_params={'vis_level': -1, 'line_thickness': 500},
            patch_params={'use_padding': True, 'contour_fn': 'four_pt'},
            patch_level=self.patch_level,
            use_default_params=False, 
            seg=True,
            save_mask=False,
            stitch=False,
            patch=True,
            auto_skip=True, 
            process_list=None,
            )


class Coord2Data:

    def __init__(
        self,
        slide_dir,
        coord_dir,
        transforms,
        exclude=None,
        csv_path=None,
        ):

        self.slide_dir = slide_dir
        self.coord_dir = coord_dir
        self.transforms = transforms
        self.csv_path = csv_path

        if csv_path is not None:
            self.df = pd.read_csv(csv_path)
            
        if exclude is not None:
            self.patient_id = [file_name.split('-')[2] for file_name in exclude]
        else:
            self.patient_id = []

    def get_dataset(self):
        ''' creates a dataset for each WSI (a bag) 
            and appends them to a list of datasets.
            if labels are available in a csv file,
            a list of labels will also be returned
        '''

        datasets = []
        labels = []
        dictionary = {}

        for file_name in os.listdir(self.slide_dir):
            try:
                
                wsi = openslide.open_slide(self.slide_dir + file_name)

                slide_id = file_name.rsplit('.', 1)[0]
                file_path = self.coord_dir + slide_id + '.h5'
                
                if file_name.split('-')[2] not in self.patient_id:
            
                    dataset = Whole_Slide_Bag_FP(
                        file_path=file_path,
                        wsi=wsi,
                        pretrained=False,
                        custom_transforms=self.transforms,
                        custom_downsample=1,
                        target_patch_size=-1,
                        )

                    #datasets.append(dataset)
                    dictionary.update({slide_id : dataset})  

                    if self.csv_path is not None:
                        label = self.df[labels['slide_id'] == slide_id]['label']
                        labels.append(label)

                        return (datasets, labels)
           
            except Exception:
                pass 

        return dictionary