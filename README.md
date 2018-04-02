# Deploying a Digits Model on a TX1 using multiple cameras for detection

**Recommended System Requirements**

Training GPU:  Maxwell or Pascal-based GPU or AWS P2 instance.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ubuntu 14.04 x86_64 or Ubuntu 16.04 x86_64 (see DIGITS [AWS AMI](https://aws.amazon.com/marketplace/pp/B01LZN28VD) image).

Deployment:    &nbsp;&nbsp;Jetson TX2 Developer Kit with JetPack 3.0 or newer (Ubuntu 16.04 aarch64).  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Jetson TX1 Developer Kit with JetPack 2.3 or newer (Ubuntu 16.04 aarch64).

> **note**:  this [branch](http://github.com/dusty-nv/jetson-inference) is verified against the following BSP versions for Jetson TX1/TX2: <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX2 - JetPack 3.1 / L4T R28.1 aarch64 (Ubuntu 16.04 LTS) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 3.1 / L4T R28.1 aarch64 (Ubuntu 16.04 LTS) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX2 - JetPack 3.0 / L4T R27.1 aarch64 (Ubuntu 16.04 LTS) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 2.3 / L4T R24.2 aarch64 (Ubuntu 16.04 LTS) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 2.3.1 / L4T R24.2.1 aarch64 (Ubuntu 16.04 LTS)

Note that TensorRT samples from the repo are intended for deployment on embedded Jetson TX1/TX2 module, however when cuDNN and TensorRT have been installed on the host side, the TensorRT samples in the repo can be compiled for PC.

## DIGITS Workflow

New to deep neural networks (DNNs) and machine learning?  Take this [introductory primer](docs/deep-learning.md) on training and inference.

<a href="https://github.com/dusty-nv/jetson-inference/blob/master/docs/deep-learning.md"><img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/digits-samples.jpg" width="800"></a>

Using NVIDIA deep learning tools, it's easy to **[Get Started](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md)** training DNNs and deploying them into the field with high performance.  Discrete GPUs are typically used in a server, PC, or laptop for training with DIGITS, while Jetson and integrated GPU is used by embedded form factors.

<a href="https://github.com/dusty-nv/jetson-inference/blob/master/docs/deep-learning.md"><img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/digits-workflow.jpg" width="700"></a>

NVIDIA [DIGITS](https://github.com/NVIDIA/DIGITS) is used to interactively train network models on annotated datasets in the cloud or PC, while TensorRT and Jetson are used to deploy runtime inference in the field. TensorRT uses graph optimizations and half-precision FP16 support to more than double DNN inferencing.  Together, DIGITS and TensorRT form an effective workflow for developing and deploying deep neural networks capable of implementing advanced AI and perception. 

## System Setup

During this tutorial, we will use a host PC (or AWS), for training DNNs, alongside a Jetson for inference.  The host PC will also serve to flash the Jetson with the latest JetPack.  First we'll setup and configure the host PC with the required OS and tools.


### Running JetPack on the Host

Download the latest **[JetPack](https://developer.nvidia.com/embedded/jetpack)** to the host PC.  In addition to flashing the Jetson with the latest Board Support Package (BSP), JetPack automatically installs tools for the host like CUDA Toolkit.  See the JetPack [Release Notes](https://developer.nvidia.com/embedded/jetpack-notes) for the full list of features and installed packages.

After downloading JetPack from the link above, run it from the host PC with the following commands:

``` bash 
$ cd <directory where you downloaded JetPack>
$ chmod +x JetPack-L4T-3.1-linux-x64.run 
$ ./JetPack-L4T-3.1-linux-x64.run 
```

The JetPack GUI will start.  Follow the step-by-step **[Install Guide](http://docs.nvidia.com/jetpack-l4t/index.html#developertools/mobile/jetpack/l4t/3.0/jetpack_l4t_install.htm)** to complete the setup.  Near the beginning, JetPack will confirm which generation Jetson you are developing for.

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/jetpack-platform.png" width="450">

Select Jetson TX1 if you are using TX1, or Jetson TX2 if you're using TX2, and press `Next` to continue.

The next screen will list the packages available to be installed.  The packages installed to the host are listed at the top under the `Host - Ubuntu` dropdown, while those intended for the Jetson are shown near the bottom.  You can select or deselect an individual package for installation by clicking it's `Action` column.

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/jetpack-downloads.png" width="500">

Since CUDA will be used on the host for training DNNs, it's recommended to select the Full install by click on the radio button in the top right.  Then press `Next` to begin setup.  JetPack will download and then install the sequence of packages.  Note that all the .deb packages are stored under the `jetpack_downloads` subdirectory if you are to need them later.  

After the downloads have finished installing, JetPack will enter the post-install phase where the JetPack is flashed with the L4T BSP.  You'll need to connect your Jetson to your host PC via the micro-USB port and cable included in the devkit.  Then enter your Jetson into recovery mode by holding down the Recovery button while pressing and releasing Reset.  If you type `lsusb` from the host PC after you've connected the micro-USB cable and entered the Jetson into recovery mode, you should see the NVIDIA device come up under the list of USB devices.  JetPack uses the micro-USB connection from the host to flash the L4T BSP to the Jetson.  

After flashing, the Jetson will reboot and if attached to an HDMI display, will boot up to the Ubuntu desktop.  After this, JetPack connects to the Jetson from the host via SSH to install additional packages to the Jetson, like the ARM aarch64 builds of CUDA Toolkit, cuDNN, and TensorRT.  For JetPack to be able to reach the Jetson via SSH, the host PC should be networked to the Jetson via Ethernet.  This can be accomplished by running an Ethernet cable directly from the host to the Jetson, or by connecting both devices to a router or switch — the JetPack GUI will ask you to confirm which networking scenario is being used.  See the JetPack **[Install Guide](http://docs.nvidia.com/jetpack-l4t/index.html#developertools/mobile/jetpack/l4t/3.0/jetpack_l4t_install.htm)** for the full directions for installing JetPack and flashing Jetson.

### Installing NVIDIA Driver on the Host

At this point, JetPack will have flashed the Jetson with the latest L4T BSP, and installed CUDA toolkits to both the Jetson and host PC.  However, the NVIDIA PCIe driver will still need to be installed on the host PC to enable GPU-accelerated training.  Run the following commands from the host PC to install the NVIDIA driver from the Ubuntu repo:

``` bash
$ sudo apt-get install nvidia-375
$ sudo reboot
```

Afer rebooting, the NVIDIA driver should be listed under `lsmod`:

``` bash
$ lsmod | grep nvidia
nvidia_uvm            647168  0
nvidia_drm             49152  1
nvidia_modeset        790528  4 nvidia_drm
nvidia              12144640  60 nvidia_modeset,nvidia_uvm
drm_kms_helper        167936  1 nvidia_drm
drm                   368640  4 nvidia_drm,drm_kms_helper
```

To verify the CUDA toolkit and NVIDIA driver are working, run some tests that come with the CUDA samples:

``` bash
$ cd /usr/local/cuda/samples
$ sudo make
$ cd bin/x86_64/linux/release/
$ ./deviceQuery
$ ./bandwidthTest --memory=pinned
```

### Installing cuDNN on the Host

The next step is to install NVIDIA **[cuDNN](https://developer.nvidia.com/cudnn)** libraries on the host PC.  Download the libcudnn and libcudnn packages from the NVIDIA site:

```
https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170307/Ubuntu16_04_x64/libcudnn6_6.0.20-1+cuda8.0_amd64-deb
https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170307/Ubuntu16_04_x64/libcudnn6-dev_6.0.20-1+cuda8.0_amd64-deb
```

Then install the packages with the following commands:

``` bash
$ sudo dpkg -i libcudnn6_6.0.20-1+cuda8.0_amd64.deb
$ sudo dpkg -i libcudnn6-dev_6.0.20-1+cuda8.0_amd64.deb
```

### Installing NVcaffe on the Host

NVcaffe is the NVIDIA branch of Caffe with optimizations for GPU.  NVcaffe uses cuDNN and is used by DIGITS for training DNNs.  To install it, clone the NVcaffe repo from GitHub and compile from source.  Use the NVcaffe-0.15 branch like below.

> **note**: for this tutorial, NVcaffe is only required on the host (for training).  During inferencing phase TensorRT is used on the Jetson and doesn't require caffe.

First some prequisite packages for Caffe are installed, including the Python bindings required by DIGITS:

``` bash
$ sudo apt-get install --no-install-recommends build-essential cmake git gfortran libatlas-base-dev libboost-filesystem-dev libboost-python-dev libboost-system-dev libboost-thread-dev libgflags-dev libgoogle-glog-dev libhdf5-serial-dev libleveldb-dev liblmdb-dev libprotobuf-dev libsnappy-dev protobuf-compiler python-all-dev python-dev python-h5py python-matplotlib python-numpy python-opencv python-pil python-pip python-protobuf python-scipy python-skimage python-sklearn python-setuptools 
$ sudo pip install --upgrade pip
$ git clone -b caffe-0.15 http://github.com/NVIDIA/caffe
$ cd caffe
$ sudo pip install -r python/requirements.txt 
$ mkdir build
$ cd build
$ cmake ../ -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF
$ make --jobs=4
$ make pycaffe
```

Caffe should now be configured and built.  Now edit your user's ~/.bashrc to include the path to your Caffe tree (replace the paths below to reflect your own):

``` bash
export CAFFE_ROOT=/home/dusty/workspace/caffe
export PYTHONPATH=/home/dusty/workspace/caffe/python:$PYTHONPATH
```

Close and re-open the terminal for the changes to take effect.


## Building from Source on Jetson
Provided along with this repo are TensorRT-enabled deep learning primitives for running Googlenet/Alexnet on live camera feed for image recognition, pedestrian detection networks with localization capabilities (i.e. that provide bounding boxes), and segmentation.  This repo is intended to be built & run on the Jetson and to accept the network models from the host PC trained on the DIGITS server.

The latest source can be obtained from [GitHub](http://github.com/dusty-nv/jetson-inference) and compiled onboard Jetson TX1/TX2.

> **note**:  this [branch](http://github.com/dusty-nv/jetson-inference) is verified against the following BSP versions for Jetson TX1/TX2: <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX2 - JetPack 3.0 / L4T R27.1 aarch64 (Ubuntu 16.04 LTS) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 2.3 / L4T R24.2 aarch64 (Ubuntu 16.04 LTS) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 2.3.1 / L4T R24.2.1 aarch64 (Ubuntu 16.04 LTS)
      
#### Cloning the Repo
To obtain the repository, navigate to a folder of your choosing on the Jetson.  First, make sure git and cmake are installed locally:

``` bash
$ sudo apt-get install git cmake
```

Then clone the jetson-inference repo:
``` bash
$ git clone http://github.com/dusty-nv/jetson-inference
```

#### Configuring with CMake

When cmake is run, a special pre-installation script (CMakePreBuild.sh) is run and will automatically install any dependencies.

``` bash
$ cd jetson-inference
$ mkdir build
$ cd build
$ cmake ../
```

> **note**: the cmake command will launch the CMakePrebuild.sh script which asks for sudo while making sure prerequisite packages have been installed on the Jetson. The script also downloads the network model snapshots from web services.

#### Compiling the Project

Make sure you are still in the jetson-inference/build directory, created above in step #2.

``` bash
$ cd jetson-inference/build			# omit if pwd is already /build from above
$ make
```

Depending on architecture, the package will be built to either armhf or aarch64, with the following directory structure:

```
|-build
   \aarch64		    (64-bit)
      \bin			where the sample binaries are built to
      \include		where the headers reside
      \lib			where the libraries are build to
   \armhf           (32-bit)
      \bin			where the sample binaries are built to
      \include		where the headers reside
      \lib			where the libraries are build to
```

binaries residing in aarch64/bin, headers in aarch64/include, and libraries in aarch64/lib.


	 
# Transfering model files to jetson from host
 let's download and extract the model snapshot to Jetson.  From the browser on your Jetson TX1/TX2, navigate to your DIGITS server and the `GoogleNet-ILSVRC12-subset` model.  Under the `Trained Models` section, select the desired snapshot from the drop-down (usually the one with the highest epoch) and click the `Download Model` button.

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-digits-model-download.png" width="650">

Alternatively, if your Jetson and DIGITS server aren't accessible from the same network, you can use the step above to download the snapshot to an intermediary machine and then use SCP or USB stick to copy it to Jetson.  

Then extract the archive with a command similar to:

```cd <directory where you downloaded the snapshot>
tar -xzvf 20170524-140310-8c0b_epoch_30.0.tar.gz
```

Next we will load our custom snapshot into TensorRT, running on the Jetson.


### Downloading the Model Snapshot to Jetson

Next, download and extract the trained model snapshot to Jetson.  From the browser on your Jetson TX1/TX2, navigate to your DIGITS server and the `DetectNet-COCO-Dog` model.  Under the `Trained Models` section, select the desired snapshot from the drop-down (usually the one with the highest epoch) and click the `Download Model` button.

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-model-download-dog.png" width="650">

Alternatively, if your Jetson and DIGITS server aren't accessible from the same network, you can use the step above to download the snapshot to an intermediary machine and then use SCP or USB stick to copy it to Jetson.  

Then extract the archive with a command similar to:

```cd <directory where you downloaded the snapshot>
tar -xzvf 20170504-190602-879f_epoch_100.0.tar.gz

## Extra Resources

In this area, links and resources for deep learning developers are listed:

* [Appendix](docs/aux-contents.md)
	* [NVIDIA Deep Learning Institute](https://developer.nvidia.com/deep-learning-institute) — [Introductory QwikLabs](https://developer.nvidia.com/deep-learning-courses)
     * [Building nvcaffe](docs/building-nvcaffe.md)
	* [Other Examples](docs/other-examples.md)
	* [ros_deep_learning](http://www.github.com/dusty-nv/ros_deep_learning) - TensorRT inference ROS nodes

