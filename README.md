# Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP

## Introduction
Object pose estimation has a wide range of applications in the field of robots and self-driving since it is essential to get precise orientations and translations of objects in these tasks. Methods that combined with depth map can get good results. However, depth map is not easy to get. As such, our project is dedicated to estimating poses of known objects using RGB images only.

Instead of directly predicting representations of poses, many methods solve the pose using Perspective-n-Points (PnP) through the correspondence be- tween 2D and 3D. But this has a disadvantage: the training of the network is not end-to-end. The pose estimated by PnP does not guide the update of front-end network parameters. The key problem is that when we use back-propagation, the gradient has no way to pass through PnP. In this project, we originally combined a backpropagatable PnP method in the training framework and introduced a projection loss, so as to achieve an end-to-end training.

Another significant challenge in pose estimation is to estimate correct poses when objects are occluded or symmet- ric. In this project, we simply ignore the rotation relative to the axis on which infinite and continuous symmetric poses exist. For the symmetric objects that have a finite number of ambiguous views, we adopt the transformer loss to handle it. We also use synthesized images and data augmen- tations to generate more train data and relieve the problem of occlusions.

To summarize, the main work of this project are: (1) Im- plement and train the Mask-RCNN for object detection and instance segmentation. (2) Implement the main framework of Pix2Pose in Pytorch and achieve the reported per- formance. (3) One major novelty of this work is through combining the backpropagatable PnP method, we achieve an end-to-end training process and get a better performance.

## Our Methods
This section provides a detailed description of our net- work architecture and loss functions for training. The tar- get of our method is to predict R and T for each object, which describes the rotation and translation of cameras in world coordinate system.

![figure1](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/figure1.png)

![figure2](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/figure2.png)

![figure3-4](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/figure3-4.png)

### Object Detection
In the first stage of the pipeline, an independent Mask- RCNN is employed for 2-D object detection and in- stance segmentation. This stage takes the whole RGB image as input and will provide classes, 2-D bounding boxes and instance masks for detected objects. RGB images as well as masks of these detected objects will then be cropped according to enlarged 2-D bounding boxes in order to deal with possible occlusions. These cropped images will then be processed by lateral Unet.

### 3-D Coordinate Prediction
The architecture of the our network is described in Fig.3. We use a Unet with skip connection network architecture to make a pixel-to-pixel prediction of 3-D coor- dinates in object coordinate system.

![equation1](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/equation1.png)

The input of the network is cropped images from the re- sults of Mask-RCNN, Iinput. The outputs of the network are normalized 3-D coordinates of each pixel in the object coordinate system, I3D, and the error level map of pixels, Ie. The ground truth Igt is a map with same size as input image Iinput and three channels. These three channels de- note the x, y, z coordinate of each pixel of Iinput in object coordinate system. A sample pair of Iinput and Igt is shown in Fig.4. Since Igt can also be seen as a image, we can gen- erate Igt by rendering 3-D model of a object with its 3-D coordinate encoded as vertex colors with same pose as its ground truth.

The target of training is to predict the target coordinate image from an input image. To evaluate the performance of the prediction, a comprehension loss function is intro- duced. The first part of the loss is called transformer loss, Ltrans, which will be illustrated in this section. The next section will elaborate the another part of loss function, the projection loss, Lproj.

To make the prediction more close to Igt, the average L1 distance of each pixel loss is used. The construction of Lr is defined as:

![equation2](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/equation2.png)

In which n is the number of pixels. M is an object mask containing all pixels
that belong to the object when it is fully observable. This mask makes the loss which has the motivation of predicting the occluded part of object. We give the factor β some value >= 1, meaning that the pixel belongs to the object is more significant than the pixel that does not.

We have talked about the difficulty of handling the sym- metric object. Since such an object can have a huge pixel distance but a very similar pose to a certain pose and will make neural network hard to converge. One solution is to multiple the 3D transform matrix in the symmetric pool to the Igt, so that these 3-D coordinates can be transformed to coordinates in another symmetric pose. Then one of the pose that has the smallest loss error can be chose, deriving L3D as follow:

![equation3](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/equation3.png)

where Rp is the transforming matrix from the pose of Igt to a symmetric pose p. sym is a pool of all symmetric pose and is a pre-defined parameter for each object.

Another information we want in the output is the error map of each pixel, which estimates the difference between I3D and Igt and is used to refine the output image during inference. Ie should be identical to the Lr with β = 1. Based on that the error prediction loss have the form:

![equation4](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/equation4.png)

Lr is clipped to the maximum output of sigmoid function. With the above loss function, the transform loss Ltrans can be written as:

![equation5](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/equation5.png)

where λ1 and λ2 denote weights to balance different tasks.

### Backpropagatable PnP
In this section, we define the projection loss Lproj to ensure projection with predicted pose can get similar pro- jection results with ground truth pose.

Let g denotes a PnP solver in the form of a mapping

![equation6](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/equation6.png)

which returns the 6 DoF pose y of a camera with intrinsic matrix K ∈ R3×3 from n 2D-3D correspondences. y can be parametrized as following:

![equation7](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/equation7.png)

In our work, we use axis-angle representation, i.e. m = 6.

Let π(·|y,K) be a projective transformation of 3-D points onto the image plane with pose y and camera intrin- sics K. The projection loss function Lproj has the form

![equation8](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/equation8.png)

where y is calculated by Eq.6, z∗ is the 3-D vertex coordi- nates of the object, x∗ is the 2-D image coordinate projected by ground truth pose. We take CNN as a function:

![equation9](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/equation9.png)

which is parametrized by θ. In order to update the net- work parameters θ, we take the derivative of Lproj w.r.t. θ:

![equation10](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/equation10.png)

where ∂y/∂z is addressed by using The Implicit Function Theorem. The implicit differentiation of ∂y/∂z follows

![equation11](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/equation11.png)

where f(x, y, z, K) = [f1, . . . , fm]T is the constructed constraint function for IFT, and fj is defined by for all j ∈ {1,...,m}:

![equation12](https://github.com/WallaceSUI/Estimate-6D-Pose-from-Single-RGB-Image-with-Backpropagatable-PnP/blob/main/figures-equations/equation12.png)
