import os
import time
import json
import torch
from scipy import ndimage
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from evalutils import SegmentationAlgorithm
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.utilities.helpers import empty_cache


class Uls23(SegmentationAlgorithm):
    def __init__(self):
        # Load configuration
        with open("config.json", 'r') as f:
            self.config = json.load(f)
        
        self.image_metadata = None  # Keep track of the metadata of the input volume
        self.id = None  # Keep track of batched volume file name for export
        self.z_size = 128  # Number of voxels in the z-dimension for each input VOI
        self.xy_size = 256  # Number of voxels in the xy-dimensions for each input VOI
        self.z_size_model = 64 # Number of voxels in the z-dimension that the model takes
        self.xy_size_model = 128 # Number of voxels in the xy-dimensions that the model takes
        self.device = torch.device("cuda")
        self.predictor = None # nnUnet predictor

    def start_pipeline(self):
        """
        Starts inference algorithm
        """
        start_time = time.time()
        
        # We need to create the correct output folder, determined by the interface, ourselves
        os.makedirs("/output/images/ct-binary-uls/", exist_ok=True)

        self.load_model()
        spacings = self.load_data()
        predictions = self.predict(spacings)
        self.postprocess(predictions)

        end_time = time.time()
        print(f"Total job runtime: {end_time - start_time}s")

    def load_model(self):
        # Set up the nnUNetPredictor
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False, # False is faster but less accurate
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False
        )
        # Initialize the network architecture, loads the checkpoint
        self.predictor.initialize_from_trained_model_folder(
            self.config["model_path"],
            use_folds=("all"),
            checkpoint_name="checkpoint_best.pth",
        )
    
    def load_data(self):
        """
        1) Loads the .mha files containing the VOI stacks in the input directory
        2) Unstacks them into individual lesion VOI's
        3) Crops to model input size for faster inference
        4) Saves cropped VOIs for prediction
        """
        # Input directory is determined by the algorithm interface on GC
        input_dir = Path("/input/images/stacked-3d-ct-lesion-volumes/")

        # Load the spacings per VOI
        with open(Path("/input/stacked-3d-volumetric-spacings.json"), 'r') as json_file:
            spacings = json.load(json_file)

        for input_file in input_dir.glob("*.mha"):
            self.id = input_file

            # Load and keep track of the image metadata
            self.image_metadata = sitk.ReadImage(input_dir / input_file)

            # Now get the image data
            image_data = sitk.GetArrayFromImage(self.image_metadata)
            for i in range(int(image_data.shape[0] / self.z_size)):
                voi = image_data[self.z_size * i:self.z_size * (i + 1), :, :]
                # Note: spacings[i] contains the scan spacing for this VOI

                # Crop to model input size (64x128x128) from original 128x256x256
                voi = voi[self.config["crop_z_start"]:self.config["crop_z_end"], 
                          self.config["crop_x_start"]:self.config["crop_x_end"], 
                          self.config["crop_y_start"]:self.config["crop_y_end"]]
                np.save(f"/tmp/voi_{i}.npy", np.array([voi])) # Add dummy batch dimension for nnUnet

        return spacings

    def predict(self, spacings):
        """
        Runs nnUnet inference on the cropped images
        :param spacings: list containing the spacing per VOI
        :return: list of numpy arrays containing the predicted lesion masks per VOI
        """
        predictions = []
        
        for i, voi_spacing in enumerate(spacings):
            # Load the 3D array from the binary file
            voi = np.load(f"/tmp/voi_{i}.npy")

            predictions.append(self.predictor.predict_single_npy_array(voi, {'spacing': voi_spacing}, None, None, False))

        return predictions

    def postprocess(self, predictions):
        """
        Runs post-processing and pads predictions back to original size.
        :param predictions: list of numpy arrays containing the predicted lesion masks per VOI
        """
        # Run postprocessing code here, for the baseline we only remove any
        # segmentation outputs not connected to the center lesion prediction
        for i, segmentation in enumerate(predictions):
            instance_mask, num_features = ndimage.label(segmentation)
            if num_features > 1:
                segmentation[instance_mask != instance_mask[
                    int(self.z_size_model / 2), int(self.xy_size_model / 2), int(self.xy_size_model / 2)]] = 0
                segmentation[segmentation != 0] = 1

            # Pad segmentations back to fit with original image size
            segmentation_pad = np.pad(segmentation, 
                    ((self.config["pad_z_before"], self.config["pad_z_after"]),  
                    (self.config["pad_x_before"], self.config["pad_x_after"]),   
                    (self.config["pad_y_before"], self.config["pad_y_after"])),
                    mode='constant', constant_values=0)

            # Convert padded segmentation back to a SimpleITK image
            segmentation_image = sitk.GetImageFromArray(segmentation_pad)

            # Update the origin to account for the padding
            original_origin = self.image_metadata.GetOrigin()
            original_spacing = self.image_metadata.GetSpacing()
            new_origin = [
                original_origin[0] - self.config["pad_x_before"] * original_spacing[0],  # Adjust for x padding
                original_origin[1] - self.config["pad_y_before"] * original_spacing[1],  # Adjust for y padding
                original_origin[2] - self.config["pad_z_before"] * original_spacing[2],  # Adjust for z padding
            ]
            segmentation_image.SetOrigin(new_origin)

            # Copy the direction and spacing from the original metadata
            segmentation_image.SetDirection(self.image_metadata.GetDirection())
            segmentation_image.SetSpacing(self.image_metadata.GetSpacing())

            # Save the updated segmentation image
            predictions[i] = sitk.GetArrayFromImage(segmentation_image)

        predictions = np.concatenate(predictions, axis=0)  # Stack predictions

        # Create mask image and copy over metadata
        mask = sitk.GetImageFromArray(predictions)
        mask.CopyInformation(self.image_metadata)

        sitk.WriteImage(mask, f"/output/images/ct-binary-uls/{self.id.name}")

if __name__ == "__main__":
    Uls23().start_pipeline()
