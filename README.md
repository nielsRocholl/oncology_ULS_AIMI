# ULS23 challenge model container

## Introduction
This repository is a ready-to-submit template for the ULS (Universal Lesion Segmentation) challenge on Grand Challenge. It packages your trained nnUNet model for automatic evaluation. Specifically, this is for models trained with a smaller input size (64×128×128). The code automatically crops challenge images to fit, runs your model, and pads results back so that they can be evaluated on the original data. This is useful, because training on smaller images is much faster (: 

## Data and other resources
Everything else to get started can be found here: https://zenodo.org/records/15355959. This zenodo link has three things: 
- fully_annotated_data.zip: The original challenge data has already been cropped for you, you can find that in this zip file. Along with being cropped, we left out the semi-annotated part of the original dataset. This is also for extra speed. Yes, you can expect a slight drop in performance because of the smaller and fewer images, but this will NOT affect your grade. We are aware of this. Also, your score for the challenge won't directly be reflected in the grade for the project: We far prefer a well-explained model that doesn't do much better than the baseline, than a model that does far better but has no proper documentation/explanations. 
- nnUNet_results.zip: These are the model weights of the baseline model that was trained on this cropped data. You can use this to test how this repo works, and to see what format we expect. Your goal in essence is to improve on this model.
- stacked_voi_sample.mha: This is a test image. If you want to try out your model, you can use this image to do it. The ULS23 challenge works with these stacked mha files, so that's why you need this specific image to see if your model works properly on GC. More info on this below. 

## How to use this repo
You can use this repo in two ways:
**For submission**: You upload this repo, along with a tar.gz file containing your model weights, to Grand Challenge to evaluate your model on that platform.
**For trying out**: You have this repo on your local computer, you add your weights to this repo, and see if everything runs properly locally by running it on one image. This is useful for debugging, because on Grand Challenge it might take a while to see your errors. It is however also possible to do this on GC, you can decide this yourself.

## Quick Start: Submit to Grand Challenge
### Prerequisites
- A trained nnUNet model (your `nnUNet_results` folder). If you're not familiar with nnUNet, please have a look at their documentation: https://github.com/MIC-DKFZ/nnUNet/tree/master
- A Grand Challenge account (free, you can sign up at [grand-challenge.org](https://grand-challenge.org)). 
- An algorithm page on Grand Challenge (which you can make by clicking on 'Add a new Algorithm' on the Algorithm tab)

### Steps to Submit
1. Zip your `nnUNet_results` folder into a `.tar.gz`.
2. Fork this repo so you have your own version.
3. Go to the page of your algorithm.
4. Upload your model tar under 'Model'.
5. Connect your version of this repo under 'Containers'.
6. Tag your repo (meaning make a release version. Grand Challenge sees these releases automatically).
7. Go to 'Try out algorithm', and see if this all works by uploading the stacked_voi_sample.mha that's on the zenodo page, along with the stacked_spacing_sample.json that's in this repo under architecture/input/. This is your test image, that functions as a debugger.
8. If you aren't getting any errors, your model is ready to be submitted to the ULS23 challenge! That can be done on the challenge page: https://uls23.grand-challenge.org/

## Optional: Local Testing with Docker

If you want to test locally before submitting, you have to use Docker locally. As mentioned before, this can also be done on GC, but locally might go a bit faster in terms of debugging. You will have to work a bit with Docker with this, so it might also be good practice for this (: 

### Prerequisites for Testing
- A trained nnUNet model.
- Docker: Download from [docker.com](https://www.docker.com/get-started).

### Setup for local testing

1. Clone this repo locally.
2. Put your nnUNet results in `architecture/nnUNet_results/`. The folder structure should look like this:
   ```
   architecture/
   └── nnUNet_results/
       └── Dataset090_ULS23_Combined/
           └── nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/
               ├── checkpoint_best.pth
               ├── checkpoint_final.pth
               ├── dataset.json
               └── ...other files...
   ```
   (Replace `Dataset090_ULS23_Combined` with your actual dataset name if different)
3. (Optional) Put the sample input image `stacked_voi_sample.mha` from the zenodo page in `architecture/input/images/stacked-3d-ct-lesion-volumes/` (the `stacked_spacing_sample.json` is already in the right place).
4. If you want to do CPU testing instead of GPU, change `device` in `process.py` from `"cuda"` to `"cpu"`.
5. Build the container:
   ```bash
   docker build --build-arg LOCAL_BUILD=true -t uls23 .
   ```
6. Run the container:
   ```bash
   docker run --rm -v C:\path\to\output:/output uls23
   ```
   Replace `C:\path\to\output` with a folder on your computer for the output.

## How It Works

This section is just some more info on what this repo actually does. Not necessary to know, but for your information. 

1. **Load Input**: Reads `.mha` images and spacing JSON.
2. **Crop**: Cuts to 64×128×128.
3. **Predict**: Runs your nnUNet model.
4. **Pad**: Adds back to original size.
5. **Save**: Outputs stacked masks.

### Customization
The `config.json` contains all the cropping and padding settings. You technically don't have to change anything here, but it's good to know.

### File Structure
- `process.py`: Inference script.
- `config.json`: Settings.
- `architecture/`: Your weights and test data.
- `Dockerfile`: For testing.
- `scripts/build.sh`: Build script.


## VERY IMPORTANT
It's always possible to get some versioning errors with nnunetv2 or python, since packages are constantly being updated. Generally, it's smart to use the same package versions as this repo uses. We highly recommend trying to upload your model weights early in the process, before having done all the training, just to be sure everything works. You could for example just take your second epoch weights and try it. This way, you prevent errors later on! 

Good luck!
