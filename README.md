# Synthetic Image Detection
[arXiv](https://arxiv.org/pdf/2407.10736) 

<p align="center">
<img src=assets/synthetic_vs_real_detector.jpg />
</p>

[**When Synthetic Traces Hide Real Content: Analysis of Stable Diffusion Image Laundering**](https://arxiv.org/pdf/2407.10736)<br/>
Sara Mandelli, Paolo Bestagini, Stefano Tubaro
[Image and Sound Processing Lab - Politecnico di Milano](http://ispl.deib.polimi.it/)

## Prerequisites
Create and activate the conda environment
```bash
conda env create -f environment.yml
conda activate synth_img_det
```
## Synthetic image detector

We define an image to be _real_ if its pixel content generally comes from a photograph; it can have undergone post-processing operations like compression, cropping or resizing, but its original content has been acquired by a digital camera sensor.

On the contrary, we say that the image is _synthetic_ if it is the result of a synthetic generation model applied to an input signal.

Download the model's weights from [this link](...) and unzip the file under the main folder
```bash
wget ...
unzip synth_vs_real_weigths.zip
```

### Test the synthetic vs real detector on a single image
Obtain the synthetic vs real score for a single image.
If the score is greater than 0, the image is likely synthetic. 
If the score is lower than 0, the image is likely real.

You can decide whether to test only the face area or the entire image. 

To test the entire image:
```bash
python test_real_vs_fake_singleimg.py --img_path $PATH_TO_TEST_IMAGE
```
To test only the face area:
```bash
python test_real_vs_fake_singleimg.py --select_face_test --img_path $PATH_TO_TEST_IMAGE
```
## Laundered image detector

Stable Diffusion (SD) Laundered images are defined as 

Download the model's weights from [this link](...) and unzip the file under the main folder
```bash
wget ...
unzip laundered_vs_fullysynth_weigths.zip
```

### Test the laundered vs fully-synthetic detector on a single image
Obtain the synthetic vs real score for a single image.
If the score is greater than 0, the image is likely synthetic. 
If the score is lower than 0, the image is likely real.

You can decide whether to test only the face area or the entire image. 

To test the entire image:
```bash
python test_real_vs_fake_singleimg.py --img_path $PATH_TO_TEST_IMAGE
```
To test only the face area:
```bash
python test_real_vs_fake_singleimg.py --select_face_test --img_path $PATH_TO_TEST_IMAGE
```

Bibtex:
```bibtex
@article{Mandelli2024,
  title={When Synthetic Traces Hide Real Content: Analysis of Stable Diffusion Image Laundering},
  author={Mandelli, Sara and Bestagini, Paolo and Tubaro, Stefano},
  journal={arXiv preprint arXiv:2407.10736},
  year={2024}
}
