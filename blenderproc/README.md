# Synthetic Table-Top Dataset Generation with BlenderProc

This folder contains config and additional modules for generating the synthetic training dataset of our paper.
It requires the SunCG as well as the ShapeNet dataset, but you can replace the respective Loader in the config if you want to use another synthetic dataset.

<p align="center">
<img src='../example_figures/training_data.png' width='800'>
<p>

### File overview

- [base_config.yaml](base_config.yaml): Template config which will be modified before running BlenderProc
- [check.py](check.py): Script to visualize rendered data
- [shapenet_objects.txt](shapenet_objects.txt): Textfile with relative paths to the used ShapeNet objects. Only ones with valid uv mapping are listed here, as we randomly change texture
- [suncg_houses.txt](suncg_houses.txt): Textfile with relative paths to SunCG houses


- [CustomObjectLoader.py](./CustomObjectLoader.py): BlenderProc extension module
- [OnFloorRemover.py](./OnFloorRemover.py): BlenderProc extension module
- [TableExtractor.py](TableExtractor.py): BlenderProc extension module

### Steps to create the synthetic dataset

1. Download [BlenderProc](https://github.com/DLR-RM/BlenderProc). The code was tested with version 1.9.0.

2. Make sure you have the SunCG dataset and the ShapeNetv2 dataset downloaded. Download cctextures (you can use {BLENDERPROC_ROOT}/scripts/download_cc_textures.py to download them).

3. Edit the paths in the [config generation script](./create_config.py), lines 12-15.

4. Edit the paths in the [base config](./base_config.yaml), lines 35 and 331.

5. Place the additional files in {BLENDERPROC_ROOT}, specifically:

    - CustomObjectLoader: {BLENDERPROC_ROOT}/src/loader/
    - OnFloorRemover: {BLENDERPROC_ROOT}/src/object/
    - TableExtractor: {BLENDERPROC_ROOT}/src/object/

6. Dataset creation happens in two phases: config generation and running BlenderProc.

   a. Run `python create_config.py` to modify the [base_config.yaml](base_config.yaml) with random objects and a random house path, among others. This also creates an output directory where the files are stored.
   b. Inside {BLENDERPROC_ROOT}, run `python run.py /path/to/created/config.yaml`. All necessary arguments are already inside the config. If successfully, this creates 10 hdf5 files in the directory where also the generated config is stored.
   
   In the paper we repeated these steps 4,000 times to obtain 40,000 samples.

7. Check data creation

   Run `python check.py /path/to/output_folder` with */path/to/output_folder* being the output path in the [config generation script](./create_config.py), line 15.

### BlenderProc config information

We select a random room with a table inside a SunCG house, and load five to twelve objects into the scene. With the help of a physics simulation these are placed realistically on a table surface.
After placing light sources and ten random camera poses we render ten stereo frames with object segmentation masks. 

For further details we refer to [our paper](https://arxiv.org/abs/2103.06796), Sec. 5A.
For more details on BlenderProc we refer to [its official documentation](https://dlr-rm.github.io/BlenderProc/).

### Misc

Note that we also provide the option to annotate the table. For this, outcomment Line 61 in [TableExtractor.py](TableExtractor.py).
Make sure you outcomment line 56 in the [dataset class](../data_io/data_loader.py) if you want to use the table annotation.
