# Powerline dataset and CycleGAN in Pytorch 

### CycleGAN with Powerline dataset

This is a project that applied CylceGAN to powerline dataset. The existing method used to detect power lines using weakly supervised learning.
However, this method had a problem that the segmented image extracted from the model required post-processing due to the presence of noise, which resulted in image loss. 

To solve this problem, we tried to apply the CycleGAN to this problem and the goal is to create a line-like image by learning the characteristics of lines. We have adopted CycleGAN because our dataset is unpaired dataset. Domain A is a mask image from VBP algorithm, and domain B consists of a line image randomly created using Bezier Curve. Generated images made of CycleGAN have the advantage of not having to go through the post-processing process of existing methods.

### Structure of CycleGAN
![Structure](https://user-images.githubusercontent.com/68531659/131769353-b116fe71-b5ca-4ede-8912-e0b5a19e0851.png)

## Getting Started

### Installation
- Clone this :

```
$ git clone https://github.com/ziiho08/CycleGAN.git
$ cd CycleGAN
```

- Train a model
```
$ python3 train.py
```
- Test a model
To use our trained model, download Saved_model file and choose model number(epochs).
```
$ python3 test.py --model_num 20
```

- Download a Powerline dataset : https://drive.google.com/drive/folders/1yF_sehTntKfiDyH3GGkj1c9IIsgm46IC?usp=sharing
## Result of CycleGAN
**CycleGAN Results on Powerline dataset**

(a) : Original 
(b) : Visual Mask
(c) : Ground Truth 
(d) : Generated 


![Result](https://user-images.githubusercontent.com/68531659/131627234-9695f4f6-354c-4cdd-a177-f9f0f0841915.png)


