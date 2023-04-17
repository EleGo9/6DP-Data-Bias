# Info
Official code for reproducing the results from *"Uncovering the Background-Induced bias in RGB
based 6-DoF Object Pose Estimation"* [PAPER (ArXiv)]()

# Instructions

## Installation:
1) Clone this repo recursively `git clone --recursive ....`
2) Create a new environment with `conda create -n EfficientPoseSaliency python==3.7`
3) Activate that environment with `conda activate EfficientPoseSaliency`
4) Install Tensorflow 1.15.0 with `conda install tensorflow-gpu==1.15.0`
5) Go to the repo dir and install the other dependencys using `pip install -r requirements.txt`
6) Compile cython modules with `python setup.py build_ext --inplace`
7) Download LineMod dataset and weights https://drive.google.com/drive/folders/1VcBLcIBhuT5MmXfE9NMrFdAk2xzF3mP5
8) If ou want to generate saliency maps for the images with the ArUco codes blocked, we provide 100 samples each for the objects 1, 5 amd 11.
Download the samples from [This Google Drive](https://drive.google.com/file/d/1pYaqYt-Q4f0JWU97DtKQ284eJESMm6j1) and extract the archive inside of the directory `Linemod_and_Occlusion` that you have downloaded in the previous step.

## Generating Saliency Maps
The provided script allows you to produce either **GradCAM** maps or **Guided Backprop** for the sub-tasks of **Rotation** or **Translation** of efficientpose.
To understand the required arguments please refer to the help included in `main.py`:
```
Usage: main.py [OPTIONS]

Options:
  -d, --dataset PATH              Root directory of the preprocessed LineMod
                                  Dataset, this should point to the dowloaded
                                  `Linemod_and_Occlusion` diirectory
  -f, --dest_folder PATH          Destination path for the produced saliency
                                  images. If omitted the images are displayed
                                  and not saved.
  -o, --object INTEGER            ID of the desired LineMOD object
  -w, --weights PATH              Directory of the `.h5` weights file for
                                  EfficientPose. Be careful to select the
                                  weight file for the right object.(i.e, `../w
                                  eights/object_0/phi_0_linemod_best_ADD-S.h5`
                                  for object 0)  [required]
  -m, --method [gradcam|backprop]
                                  Attribution method to use, either `gradcam`
                                  or `backprop`
  -t, --task [rotation|translation]
                                  EfficientPose Subtask to analize, either
                                  `rotation` or `translation`
  --noaruco                       If present, the saliency method is evaluated
                                  on the test images with ArUCO codes
                                  blocked.The subdirectory `rgb_noaruco` must
                                  be available in the dataset directory for
                                  the target object
  -c, --cuda BOOLEAN              Use cuda for inference, default=False
  --help                          Show this message and exit.
```

## Example
The following example wuill produce the `gradcam` saliency maps for object `1` on the task of `rotation`
```
python main.py -d /home/carmelo/DATASETS/Linemod_and_Occlusion/Linemod_preprocessed/data  -w ../weights/Linemod/object_1/phi_0_linemod_best_ADD.h5  -o 1 -m gradcam -t rotation
```

### Sample Results
![Sample Results](images/sample_results.jpg)

# Reference
If you use this repo in your research, please cite the following paper.
```
TBD;
```
