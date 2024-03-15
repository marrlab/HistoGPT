import openslide 

from torch.utils.data import DataLoader
from clam.create_patches import seg_and_patch
from clam.datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP

'''
Custom wrapper class for CLAM: Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images
Main code adapted from the original GitHub repository with minor changes: https://github.com/mahmoodlab/CLAM


Usage: Patch a slide in a list of slides based on its index in the list

slide_dir = './slide/'
coord_dir = './coord/'
save_dir = './save/'

slide_list = ['WSI-Nr-0.svs', 'WSI-Nr-1.svs', 'WSI-Nr-2.svs']

patch_size = 256
step_size = 256
seg_params = {'seg_level': 1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': 'none', 'exclude_ids': 'none'}
filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}

slideloader = SlideLoader(slide_list, slide_dir, coord_dir, save_dir, patch_size, step_size, seg_params, filter_params)
dataset, dataloader = slideloader.get_dataset_dataloader(0)

print('dataset length: ' +  str(len(dataset)) )
print('dataloader length: ' +  str(len(dataloader)))
'''


class SlideLoader:
    def __init__(
        self,
        slide_list,
        slide_dir,
        coord_dir,
        save_dir,
        patch_size,
        step_size,
        seg_params,
        filter_params,
    ):
        self.index = 0
        self.slide_list = slide_list
        self.slide_dir = slide_dir
        self.coord_dir = coord_dir
        self.save_dir = save_dir
        self.patch_size = patch_size
        self.step_size = step_size
        self.seg_params = seg_params
        self.filter_params = filter_params

    def get_slide(self):
        slide_name = self.slide_list[self.index]
        slide_path = self.slide_dir + slide_name  # + '.svs'
        return slide_name, slide_path

    def get_coordinates(self):
        slide_name, slide_path = self.get_slide()

        seg_and_patch(
            slides=[slide_name],
            source=self.slide_dir,
            save_dir=self.save_dir,
            patch_save_dir=self.coord_dir,
            mask_save_dir="",
            stitch_save_dir="",
            patch_size=self.patch_size,
            step_size=self.step_size,
            seg_params=self.seg_params,
            filter_params=self.filter_params,
            seg=True,
            patch=True,
            stitch=True,
        )

    def get_dataset(self):
        self.get_coordinates()
        slide_name, slide_path = self.get_slide()

        wsi = openslide.open_slide(slide_path)
        file_path = self.coord_dir + slide_name.rsplit(".", 1)[0] + ".h5"

        dataset = Whole_Slide_Bag_FP(
            file_path=file_path,
            wsi=wsi,
            pretrained=False,
            custom_downsample=1,
            target_patch_size=-1,
        )
        return dataset

    def get_dataset_dataloader(self, slide_index, batch_size=256, shuffle=False):
        self.index = slide_index
        dataset = self.get_dataset()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataset, dataloader
