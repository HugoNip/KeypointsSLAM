## Introduction
This project shows how to use matching keypoints to estimate pose of camera.
## Requirements
### OpenCV
#### Required Packages
OpenCV  
OpenCV Contrib

### Eigen Package (Version >= 3.0.0)
#### Source
http://eigen.tuxfamily.org/index.php?title=Main_Page

#### Compile and Install
```
cd [path-to-Eigen]
mkdir build
cd build
cmake ..
make 
sudo make install 
```

#### Search Installing Location
```
sudo updatedb
locate eigen3
```

default location "/usr/include/eigen3"



### g2o Package
#### Download
https://github.com/RainerKuemmerle/g2o

#### Compile and Install
```
cd [path-to-pangolin]
mkdir build
cd build
cmake ..
make 
sudo make install 
```

## Compile this Project
```
mkdir build
cd build
cmake ..
make 
```

## Run
### Keypoint Matching by using ORB features
```
./orb_cv // OpenCV
./orb_self // Craft
```
#### ORB Keypoints
![ORB_features.png](https://github.com/HugoNip/KeypointsSLAM/blob/master/results/ORB_features.png)

#### All Matches
![all_matches.png](https://github.com/HugoNip/KeypointsSLAM/blob/master/results/all_matches.png)

#### Good Matches
![good_matches.png](https://github.com/HugoNip/KeypointsSLAM/blob/master/results/good_matches.png)

### Pose Estimation by using 2d-2d Keypoints
```
./pose_estimation_2d2d
```
### Pose Estimation by using 3d-2d Keypoints
```
./pose_estimation_3d2d
```

### Pose Estimation by using 3d-3d Keypoints
```
./pose_estimation_3d3d
```

## Reference
[Source](https://github.com/HugoNip/slambook2/tree/master/ch7)
