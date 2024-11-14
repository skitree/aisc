import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import re
import os
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd

class WaterbirdsDataset(Dataset):
    def __init__(self, metadata, images_dir, transform=None, split=None, split_ratios=(0.65, 0.15, 0.10, 0.1), pairing_distribution=None, seed=None, preload=False, device='cpu'):
        
        self.metadata = metadata
        self.images_dir = images_dir
        self.transform = transform
        self.preload = preload
        self.device = device  
        self.pairing_distribution = pairing_distribution
        
        if split is not None:
            self.metadata = metadata[metadata['split'] == split]
        elif split_ratios is not None:
            #we are splitting the data ourself
            assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"
            
            # Shuffle and assign new splits
            if pairing_distribution is not None:
                self.metadata = self.shuffle_split(self.metadata, split_ratios, seed, pairing_distribution)
            else:
                self.metadata = self.shuffle_split(self.metadata, split_ratios, seed)

            if self.preload:
                self.preloaded_images = []
                self.labels = []
                self.bird_names = []
                self.places = []

                print("Preloading data...")
                for idx in tqdm(range(len(self.metadata))):
                    # Load and process each image
                    row = self.metadata.iloc[idx]
                    img_filename = row["img_filename"]
                    label = row["y"]
                    place = row["place"]

                    img_path = os.path.join(self.images_dir, img_filename)
                    try:
                        image = Image.open(img_path).convert('RGB')
                    except Exception as e:
                        print(f"Error loading {img_filename}: {e}")
                        continue  # Skip this image

                    # Apply transformations if any
                    if self.transform:
                        image = self.transform(image)
                    else:
                        image = transforms.ToTensor()(image)

                    # Ensure the image is a tensor
                    if not isinstance(image, torch.Tensor):
                        image = transforms.ToTensor()(image)

                    # Move image to the specified device
                    image = image.to(self.device)

                    # Extract bird species from the filename
                    pattern = r"\d+\.(.*?)/"
                    match = re.search(pattern, img_filename)
                    bird_name = match.group(1) if match else "Unknown"

                    # Store preloaded data
                    self.preloaded_images.append(image)
                    self.labels.append(label)
                    self.bird_names.append(bird_name)
                    self.places.append(place)

    def __len__(self):
        if self.preload:
            return len(self.preloaded_images)
        else:
            return len(self.metadata)

    def __getitem__(self, idx):
        if self.preload:
            # Retrieve preloaded data
            image = self.preloaded_images[idx]
            label = self.labels[idx]
            bird_name = self.bird_names[idx]
            place = self.places[idx]
            return image, label, bird_name, place
        else:
            row = self.metadata.iloc[idx]

            img_filename = row["img_filename"]
            label = row["y"]  # Bird label (0 for landbird, 1 for waterbird)
            place = row["place"]

            img_path = os.path.join(self.images_dir, img_filename)

            try:
                image = Image.open(img_path).convert('RGB')  # Convert to RGB
            except Exception as e:
                print(f"Error loading {img_filename}: {e}")
                return None, None, None, None

            # Extract bird species from the filename
            pattern = r"\d+\.(.*?)/"
            match = re.search(pattern, img_filename)
            bird_name = match.group(1) if match else "Unknown"

            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)

            return image, label, bird_name, place
        
    def get_split(self, new_split, **kwargs):
        """Returns the subset of the dataset for the specified split (0: train, 1: val, 2: test)."""
        split_metadata = self.metadata[self.metadata['new_split'] == new_split]
        return WaterbirdsDataset(split_metadata, self.images_dir, transform=self.transform, pairing_distribution=self.pairing_distribution, **kwargs)
    
    def check_class_background_pairings(self, split):
        """
        check distributions of pairs (label, background)
        """
        
        split_metadata = self.metadata[self.metadata['new_split'] == split]

        # Ensure the necessary columns are present
        if 'y' not in split_metadata.columns or 'place' not in split_metadata.columns:
            raise ValueError("Metadata must contain 'y' and 'place' columns to check pairings.")
        
        # Get unique combinations of class and background
        pair_counts = split_metadata.groupby(['y', 'place']).size().unstack(fill_value=0)

        # Display pairing counts and calculate distribution
        # Calculate the percentage distribution
        total_counts = pair_counts.sum().sum()
        pair_distribution = pair_counts.map(lambda x: f"{x} ({(x / total_counts * 100):.2f}%)")

        print(f"\nClass-Background Pairing Distribution (as % of total) for Split {split}:\n", pair_distribution)

        # Check for missing pairings
        missing_pairs = pair_counts[(pair_counts == 0).any(axis=1)]
        if missing_pairs.empty:
            print(f"All class-background pairings are represented in split {split}.")
        else:
            print(f"\nMissing pairings in split {split}:\n", missing_pairs)
    
    def updating_target(self, total, pairing_distribution, pairing_counts):
        target_counts = {}
        for (y, place), current_count in pairing_counts.items():
            # Convert the (y, place) tuple to a string key for pairing_distribution lookup
            key = f"{y},{place}"
            if key in pairing_distribution:
                proportion = pairing_distribution[key]
                target_counts[(y, place)] = int(proportion * total)
            else:
                print("pairing distribution incorrectly specified")

        print("Target counts:", target_counts) 
        return target_counts
    
    def shuffle_split(self, metadata, split_ratios, seed=None, pairing_distribution=None):
        """Shuffles the dataset and assigns splits, adjusting to match pairing_distribution"""
          
        if seed is not None:
            np.random.seed(seed)

        if pairing_distribution is not None:
        # Initial split into hold-out (15%) and main set (85%) for upsampling skewed datasets 
            hold_out_size = int(0.15 * len(metadata))
            hold_out = metadata.sample(n=hold_out_size, random_state=seed)
            main_set = metadata.drop(hold_out.index)

            available_hold_out = hold_out.copy()

            pairing_counts = main_set.groupby(['y', 'place']).size()
            #sort so that least is sampled first to avoid issues in matching
            pairing_counts = pairing_counts.sort_values()
#             print("Main set pairing counts:\n", pairing_counts)

            total_pairs = pairing_counts.sum()
            current_composition = (pairing_counts / total_pairs).to_numpy()
#             print("Current composition:\n", current_composition)
            
            # Compute target counts based on pairing_distribution
            target_counts = self.updating_target(total_pairs, pairing_distribution, pairing_counts) 

            for iteration in range(5):
                pairing_counts = main_set.groupby(['y', 'place']).size().sort_values()
                adjustments_made = False

                for (y, place), current_count in pairing_counts.items():
                    target = target_counts.get((y, place))
                    group_data = main_set[(main_set['y'] == y) & (main_set['place'] == place)]

                    #get hold out with place and bird groupings
                    hold_out_group_data = available_hold_out[(available_hold_out['y'] == y) & (available_hold_out['place'] == place)]

                    if target > current_count:
                        # Attempt to upsample from hold-out
                        num_samples_needed = target - current_count
#                         print(f"needed for {y,place}", num_samples_needed)

                        if len(hold_out_group_data) >= num_samples_needed:
                            additional_data = hold_out_group_data.sample(n=num_samples_needed, random_state=seed, replace=False)
                            available_hold_out = available_hold_out.drop(additional_data.index)
                            group_data = pd.concat([group_data, additional_data])
                            adjustments_made = True

#                             print("enough samples for update")
                        else:
                            # Not enough samples in hold-out, adjust the target for this pairing
#                             print(f"Not enough data in hold-out for {(y, place)}. Adjusting target count.")
                            num_samples_added = len(hold_out_group_data)
                            group_data = pd.concat([group_data, hold_out_group_data])

                            #update target
                            temp_target_total = len(group_data) / pairing_distribution[f"{y},{place}"]
                            target_counts = self.updating_target(temp_target_total, pairing_distribution, pairing_counts)
#                             print(f"new target total counts", target_counts)
                            adjustments_made = True

                    elif target < current_count:
                        # Downsample if needed
                        if target > 0:
                            group_data = group_data.sample(n=target, random_state=seed, replace=False)
                            adjustments_made = True

                    # Replace the adjusted group data in the main set
                    main_set = main_set.drop(main_set[(main_set['y'] == y) & (main_set['place'] == place)].index)
                    main_set = pd.concat([main_set, group_data])

                # Check if adjustments were made
                if not adjustments_made:
                    print("No further adjustments.")
                    break

            # Final composition check
            updated_counts = main_set.groupby(['y', 'place']).size()
            updated_composition = (updated_counts / updated_counts.sum()).to_numpy()

            print("Final adjusted main set pairing counts:\n", main_set.groupby(['y', 'place']).size())
            main_set = main_set.sample(frac=1, random_state=seed).reset_index(drop=True)
            # After achieving the desired proportions, shuffle and stratify into four splits
        else:
            print("Pairing_distribution not specified. Data will be skewed.")
            main_set = metadata.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        total_samples = len(main_set)
        train_size = int(total_samples * split_ratios[0])
        val_size = int(total_samples * split_ratios[1])
        test_size = int(total_samples * split_ratios[2])
        ft_size = total_samples - train_size - val_size - test_size

        # Assign new splits, ignoring any existing 'split' column
        main_set['new_split'] = -1
        main_set.loc[:train_size - 1, 'new_split'] = 0  # Train
        main_set.loc[train_size:train_size + val_size - 1, 'new_split'] = 1  # Validation
        main_set.loc[train_size + val_size:train_size + val_size + test_size - 1, 'new_split'] = 2  # Test
        main_set.loc[train_size + val_size + test_size:, 'new_split'] = 3  # Fine-tuning 
            
        return main_set
