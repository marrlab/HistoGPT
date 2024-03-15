""" 
Whole Slide Image Patch Feature Extractor
Author: Valentin Koch / Helmholtz Munich
"""

import cv2
import concurrent.futures
import h5py
import json
import time

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple, List, Dict, Optional

import math
import numpy as np
import pandas as pd
import slideio
import torch
import torchvision
import torchvision.transforms as T

from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader


@dataclass
class PatchingConfigs:
    #device: str = 'cuda'
    batch_size: int = 256

    patch_size: int = 512
    downscaling_factor: float = 0.0
    resolution_in_mpp: float = 1.0

    slide_path: str = 'PATH'
    save_path: str = 'PATH'
    file_extension: str = '.ndpi'

    save_patch_images: bool = False
    save_tile_preview: bool = False

    white_thresh: list = field(default_factory=lambda: [175, 190, 178])
    black_thresh: list = field(default_factory=lambda: [0, 0, 0])
    calc_thresh: list = field(default_factory=lambda: [40, 40, 40])
    invalid_ratio_thresh: float = 0.3
    edge_threshold: int = 1

    split: list[int] = field(default_factory=lambda: [0, 1])
    exctraction_list: str = None

    @classmethod
    def from_json(cls, file):
        with open(file, "r") as f:
            return cls(**json.load(f))


class SlideDataset(Dataset):
    def __init__(
        self,
        slide: slideio.py_slideio.Slide,
        coordinates: pd.DataFrame,
        patch_size: int = 512,
        transform: list = None
    ):
        super(SlideDataset, self).__init__()
        self.slide = slide
        self.coordinates = coordinates
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        entry = self.coordinates.iloc[idx]
        x = entry['x']
        y = entry['y']

        patch = self.slide[x:x + self.patch_size, y:y + self.patch_size, :]
        img = Image.fromarray(patch)  # Convert image to RGB

        if self.transform:
            img = self.transform(img)

        return img


def get_models(device):
    """
    ToDo: add loop that iterates over a list of models and adds them to model_dicts
    """
    model_dicts = []
    model_name = 'resnet18'

    model = torchvision.models.resnet18()
    model.to(device)
    model.eval()

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    size = 224

    transforms = T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    model_dicts.append(
        {
            'name': model_name,
            'model': model.to(device),
            'transforms': transforms
        }
    )

    return model_dicts


def get_driver(extension_name: str):
    """
    Determine the driver to use for opening an image file based on its extension.
    """
    if extension_name in [".tiff", ".tif", ".jpg", ".jpeg", ".png"]:
        return "GDAL"
    elif extension_name == "":
        return "DCM"
    else:
        return extension_name.replace(".", "").upper()


def get_scaling(args: NamedTuple, mpp_resolution_slide: float):
    """
    Determine the scaling factor to apply to a whole slide image.
    """
    if args.downscaling_factor > 0:
        return args.downscaling_factor
    else:
        return args.resolution_in_mpp / (mpp_resolution_slide * 1e06)


def threshold(patch: np.array, args: NamedTuple):
    """
    Returns a boolean value indicating whether the patch is valid or not.
    """

    # Count the number of whitish pixels in the patch
    whitish_pixels = np.count_nonzero(
        (patch[:, :, 0] > args.white_thresh[0]) &
        (patch[:, :, 1] > args.white_thresh[1]) &
        (patch[:, :, 2] > args.white_thresh[2])
    )

    # Count the number of black pixels in the patch
    black_pixels = np.count_nonzero(
        (patch[:, :, 0] <= args.black_thresh[0]) &
        (patch[:, :, 1] <= args.black_thresh[1]) &
        (patch[:, :, 2] <= args.black_thresh[2])
    )
    dark_pixels = np.count_nonzero(
        (patch[:, :, 0] <= args.calc_thresh[0]) &
        (patch[:, :, 1] <= args.calc_thresh[1]) &
        (patch[:, :, 2] <= args.calc_thresh[2])
    )
    calc_pixels = dark_pixels - black_pixels

    # We always want to keep calc in!
    if (calc_pixels / (patch.shape[0] * patch.shape[1]) >= 0.05):
        return True

    # Compute the ratio of foreground pixels to total pixels in the patch
    invalid_ratio = (whitish_pixels + black_pixels) / (patch.shape[0] * patch.shape[1])

    # Check if the ratio exceeds the threshold for invalid patches
    if invalid_ratio <= args.invalid_ratio_thresh:
        # Compute the edge map of the patch using Canny edge detection
        edge = cv2.Canny(patch, 40, 100)

        # If the maximum edge value is greater than 0, compute the
        # mean edge value as a percentage of the maximum value
        if np.max(edge) > 0:
            edge = np.mean(edge) * 100 / np.max(edge)
        else:
            edge = 0

        # Check if the edge value is below the threshold for invalid patches or is NaN
        if (edge < args.edge_threshold) or np.isnan(edge):
            return False
        else:
            return True

    else:
        return False


def save_tile_preview(args, slide_name, scn, wsi, coords, preview_path):
    """
    Save the tile preview image with the specified size.
    """

    # Draw bounding boxes for each tile on the whole slide image
    def draw_rect(wsi, x, y, size, color=[0, 0, 0], thickness=4):
        x2, y2 = x + size, y + size
        wsi[y:y + thickness, x:x + size, :] = color
        wsi[y:y + size, x:x + thickness, :] = color
        wsi[y:y + size, x2 - thickness:x2, :] = color
        wsi[y2 - thickness:y2, x:x + size, :] = color

    for _, [scene, x, y] in coords.iterrows():
        if scn == scene:
            draw_rect(wsi, y, x, args.patch_size)

    # Convert NumPy array to PIL Image object
    preview_im = Image.fromarray(wsi)

    # Determine new dimensions of the preview image while maintaining aspect ratio
    preview_size = int(args.preview_size)
    width, height = preview_im.size
    aspect_ratio = height / width

    if aspect_ratio > 1:
        new_height = preview_size
        new_width = int(preview_size / aspect_ratio)
    else:
        new_width = preview_size
        new_height = int(preview_size * aspect_ratio)

    # Resize the preview image
    preview_im = preview_im.resize((new_width, new_height))

    # Save the preview image to disk
    preview_im.save(preview_path / f"{slide_name}_{scn}.png")


def save_hdf5(
    args: NamedTuple,
    slide_name: str,
    coords: pd.DataFrame,
    feats: Dict,
    slide_sizes: List[tuple],
):
    """
    Save the extracted features and coordinates to an HDF5 file.
    """
    for model_name, features in feats.items():
        if len(features) > 0:
            with h5py.File(
                Path(args.save_path) / "h5_files" / (
                    f"{args.patch_size}px_"
                    f"{model_name}_"
                    f"{args.resolution_in_mpp}mpp_"
                    f"{args.downscaling_factor}xdown_"
                    "normal"
                ) / f"{slide_name}.h5", "w"
            ) as f:
                f["coords"] = coords.astype("float64")
                f["feats"] = features
                #f["args"] = json.dumps(vars(args))
                f["model_name"] = model_name
                f["slide_sizes"] = slide_sizes

            if len(np.unique(coords.scn)) != len(slide_sizes):
                print(
                    "SEMIWARNING, at least for one scene of ",
                    slide_name,
                    "no features were extracted, reason could be poor slide quality.",
                )
        else:
            print(
                "WARNING, no features extracted at slide",
                slide_name,
                "reason could be poor slide quality.",
            )


def process_row(wsi: np.array, scn: int, x: int, args: NamedTuple, slide_name: str):
    """
    Process a row of a slide and return the coordinates of patches that are valid.
    """
    patches_coords = pd.DataFrame()

    for y in range(0, wsi.shape[1], args.patch_size):
        # check if a full patch still 'fits' in y direction
        if y + args.patch_size > wsi.shape[1]:
            continue

        # extract patch
        patch = wsi[x:x + args.patch_size, y:y + args.patch_size, :]

        # check threhold criteria
        if threshold(patch, args):
            # whether to save image patches
            if args.save_patch_images:
                img = Image.fromarray(patch)
                img.save(
                    Path(args.save_path) / "patches" / str(args.downscaling_factor) /
                    slide_name / f"{slide_name}_patch_{scn}_{x}_{y}.png"
                )
            # append extracted coordinates
            coords = pd.DataFrame({"scn": [scn], "x": [x], "y": [y]})
            patches_coords = pd.concat([patches_coords, coords], ignore_index=True)

    return patches_coords


def patches_to_feature(
    wsi: np.array,
    coords: pd.DataFrame,
    model_dicts: List[Dict],
    device: torch.device,
    args: NamedTuple,
):
    patches_features = {model_dict["name"]: [] for model_dict in model_dicts}

    with torch.no_grad():
        for model_dict in model_dicts:
            model = model_dict["model"]
            transform = model_dict["transforms"]
            model_name = model_dict["name"]

            dataset = SlideDataset(wsi, coords, args.patch_size, transform)
            dataloader = DataLoader(
                dataset, batch_size=args.batch_size, num_workers=0, shuffle=False
            )

            for batch in dataloader:
                batch = batch.to(device)
                features = model(batch.float())
                patches_features[model_name] += (features.cpu().numpy().tolist())

    return patches_features


def extract_features(
    slide: slideio.py_slideio.Slide,
    slide_name: str,
    model_dicts: List[Dict],
    device: torch.device,
    preview_path: Optional[str],
    args: NamedTuple,
):
    """
    Patch a slide and extract its features using a given model.
    """

    slide_feats = {model_dict["name"]: [] for model_dict in model_dicts}
    slide_coords = pd.DataFrame({"scn": [], "x": [], "y": []}, dtype=int)

    if args.save_patch_images:
        (Path(args.save_path) / "patches" / str(args.downscaling_factor) /
         slide_name).mkdir(parents=True, exist_ok=True)

    slide_sizes = []
    # iterate over scenes of the slides
    for scn in range(slide.num_scenes):
        scene = slide.get_scene(scn)
        slide_sizes.append(scene.size)
        scene_coords = pd.DataFrame({"scn": [], "x": [], "y": []}, dtype=int)

        # read the scene in the desired resolution
        scaling = get_scaling(args, scene.resolution[0])
        wsi = scene.read_block(
            size=(int(scene.size[0] // scaling), int(scene.size[1] // scaling))
        )
        #"""
        # Define the main loop that processes all patches
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            linspace = range(0, wsi.shape[0], args.patch_size)
            desc = slide_name + "_" + str(scn)

            # iterate over x (width) of scene
            for x in tqdm(linspace, position=1, leave=False, desc=desc):
                # check if a full patch still 'fits' in x direction
                if x + args.patch_size > wsi.shape[0]:
                    continue
                future = executor.submit(process_row, wsi, scn, x, args, slide_name)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                patches_coords = future.result()
                if len(patches_coords) > 0:
                    scene_coords = pd.concat(
                        [scene_coords, patches_coords], ignore_index=True
                    )
            #"""
        """
        linspace = range(0, wsi.shape[0], args.patch_size)
        desc = slide_name + "_" + str(scn)
        # Iterate over x (width) of scene
        for x in tqdm(linspace, position=1, leave=False, desc=desc):
            # Check if a full patch still 'fits' in x direction
            if x + args.patch_size > wsi.shape[0]:
                continue
            # Directly call the process_row function
            patches_coords = process_row(wsi, scn, x, args, slide_name)
            if len(patches_coords) > 0:
                scene_coords = pd.concat(
                    [scene_coords, patches_coords], ignore_index=True
                )
         """

        slide_coords = pd.concat([slide_coords, scene_coords], ignore_index=True)

        if len(model_dicts) > 0:
            scene_feats = patches_to_feature(
                wsi, scene_coords, model_dicts, device, args
            )
            for key in scene_feats.keys():
                slide_feats[key].extend(scene_feats[key])

        # saves tiling preview on slide in desired size
        if args.save_tile_preview:
            save_tile_preview(args, slide_name, scn, wsi, slide_coords, preview_path)

    # Write data to HDF5
    if len(model_dicts) > 0:
        save_hdf5(args, slide_name, slide_coords, slide_feats, slide_sizes)


def main(args):
    # Set device to GPU if available, else CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Get slide files based on the provided path and file extension
    slide_files = sorted(Path(args.slide_path).glob(f"**/*{args.file_extension}"))

    # Use only a subset of the slides whose names are contained in a .csv file.
    if bool(args.exctraction_list) is not False:
        to_extract = pd.read_csv(args.exctraction_list).iloc[:, 0].tolist()
        slide_files = [file for file in slide_files if file.name in to_extract]

    # (k,n): Split slides into n distinct chunks and process number k. Use (0,1) for all
    # slides at once. For example, one run with (0,2) and one with (1,2) to split data.
    chunk_len = math.ceil(len(slide_files) / args.split[1])
    start = args.split[0] * chunk_len
    end = min(start + chunk_len, len(slide_files))
    slide_files = slide_files[start:end]

    # Get model dictionaries
    model_dicts = get_models(device)

    # Get the driver for the slide file extension
    driver = get_driver(args.file_extension)

    # Create output directory
    output_path = Path(args.save_path) / "h5_files"
    output_path.mkdir(parents=True, exist_ok=True)

    # Process models
    for model in model_dicts:
        model_name = model["name"]
        save_dir = (
            Path(args.save_path) / "h5_files" /
            f"{args.patch_size}px_{model_name}_{args.resolution_in_mpp}mpp_{args.downscaling_factor}xdown_normal"
        )

        # Create save directory for the model
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create a dictionary of argument names and values
        arg_dict = vars(args)

        # Write the argument dictionary to a text file
        with open(save_dir / "config.yml", "w") as f:
            for arg_name, arg_value in arg_dict.items():
                f.write(f"{arg_name}: {arg_value}\n")

    # Create directories
    if args.save_tile_preview:
        preview_path = (
            Path(args.save_path) /
            f"tiling_previews_{args.patch_size}px_{args.resolution_in_mpp}mpp_{args.downscaling_factor}xdown_normal"
        )
        preview_path.mkdir(parents=True, exist_ok=True)
    else:
        preview_path = None

    # Process slide files
    start = time.perf_counter()
    for slide_file in tqdm(slide_files, position=0, leave=False, desc="slides"):
        slide = slideio.Slide(str(slide_file), driver)
        slide_name = slide_file.stem
        extract_features(slide, slide_name, model_dicts, device, preview_path, args)

    end = time.perf_counter()
    elapsed_time = end - start

    print("Time taken: ", elapsed_time, "seconds")


if __name__ == "__main__":
    args = PatchingConfigs()
    main(args)
