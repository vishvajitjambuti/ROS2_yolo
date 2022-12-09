# **Install ROS2 and PyTorch for yolov5 On Jetson Nano** 

This Document is useful to install **ROS2**, on **Jetson Nano** with **CUDA enabled pytorch and openCV**

- fist we need to **upgrade  Jetson nano** from 18.04 to 20.04 and the install **ros foxy** and **Install pytorch** 


## System requirement

- Jetson Nano 
- Jetpack 4.6 or higher 

## Jetpack Setup 

```python
free -m

```         
If you don't have the right amount of swap, or want to change the value, use the following procedure to do so (from a terminal):


```python
# Disable ZRAM:
sudo systemctl disable nvzramconfig

# Create 4GB swap file
sudo fallocate -l 4G /mnt/4GB.swap
sudo chmod 600 /mnt/4GB.swap
sudo mkswap /mnt/4GB.swap

# Append the following line to /etc/fstab
sudo su
echo "/mnt/4GB.swap swap swap defaults 0 0" >> /etc/fstab
exit
```
# REBOOT!
               



## **Upgrade Jetpack OS from Ubuntu 18.04 to 20.04** 

- Officially Jetpack only supports Ubuntu 18.04. In order to install ros2 foxy distribution  we need to upgrade the system from 18.04 to 20.04. 

- Further support wont be provided from nvidia and officially I guess

### Steps to Upgrade ubuntu 18.04 to Ubuntu 20.04 LTS on Jetson Nano

#### step 1 

- By default, the release upgrade feature is disabled on the provided ubuntu 18.04, to re-enable it, we need to modify the following file:

```python
sudo vi /etc/update-manager/release-upgrades

```

- And change the value of Prompt as follow: 

```python
Prompt=lts

```

- here On Jetpack default value (Prompt=Never)
- Or sometime after changing this value to lts it will ask to change it to normal (Prompt=normal)::**Dont Do it**

#### Step 2 

- The system need to be updated before the release upgrade:

```python
sudo apt update && sudo apt upgrade

```

#### Step 3 

- Before going to Next step You need to uninstall and remove chromium browser package 
- to avoid problem arise durning upgrade

```python
sudo dpkg --remove chromium-browser
sudo dpkg --remove chromium-browser-l10n

```

#### Step 4 

- After the update, the release upgrade can be performed by the following command:

```python
sudo do-release-upgrade -d -f DistUpgradeViewGtk3

```

This Step takes around 60 to 120 minitas 

#### Step 5  After the upgrade 

- Remove remove the old unity desktop, since ubuntu 20.04 replaced it with Gnome 3: 


```python
sudo apt purge unity
sudo apt autoremove

```

- Disable the annoying system error report by modify : 


```python
sudo vi /etc/default/apport

```

- and set it 

```python
enabled=0

```

#### Step 6 After the upgrade 

- Clean up unused packages:

```python
sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*
sudo apt-get remove --purge transmission*
 
sudo apt-get clean
sudo apt-get autoremove

```

#### Step 7 After the upgrade 

- It is also important to re-enable NVIDIA repositories which hosts hardware specified packages such as firmwares:
- run it in terminal 

```python
for f in /etc/apt/sources.list.d/*; do
  sudo sed -i 's/^\#\s*//' $f
done

```

#### Step 8 After the upgrade 

- Remove unused service 

```python
sudo systemctl disable containerd.service
sudo systemctl disable cups.service
sudo systemctl disable cups-browsed.service
sudo systemctl disable  avahi-daemon.service
# if you use lightdm
sudo systemctl disable lightdm
# disable snapd
sudo systemctl mask snapd.seeded.service
sudo systemctl mask snapd.socket
sudo systemctl mask snapd
# account daemon
sudo systemctl mask  accounts-daemon.service
# ubuntu error report
sudo systemctl mask whoopsie.service
# file indexing
systemctl  mask tracker-{miner-apps,miner-fs,store}
gsettings set org.freedesktop.Tracker.Miner.Files enable-monitors false
 
# disable the system to go to sleep, this is usefull in headless mode
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```


## **PyTorch and TorchVision install  on Nano** 

- PyTorch version 1.12.0 
- TorchVision 0.13.0 

- This pytorch wheel is built by Qengineering/PyTorch-Jetson-Nano this is link to  github page[link](https://github.com/Qengineering/PyTorch-Jetson-Nano)

- If you want to install different version of PyTorch and TorchVision check this the link by Qengineering[link](https://qengineering.eu/install-pytorch-on-jetson-nano.html) 

### PyTorch install


- Only for a Jetson Nano with Ubuntu 20.04

```python

# install the dependencies (if not already onboard)
$ sudo apt-get install python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
$ sudo -H pip3 install future
$ sudo pip3 install -U --user wheel mock pillow
$ sudo -H pip3 install testresources
# above 58.3.0 you get version issues
$ sudo -H pip3 install setuptools==58.3.0
$ sudo -H pip3 install Cython
# install gdown to download from Google drive
$ sudo -H pip3 install gdown
# download the wheel
$ gdown https://drive.google.com/uc?id=1MnVB7I4N8iVDAkogJO76CiQ2KRbyXH_e
# install PyTorch 1.12.0
$ sudo -H pip3 install torch-1.12.0a0+git67ece03-cp38-cp38-linux_aarch64.whl
# clean up
$ rm torch-1.12.0a0+git67ece03-cp38-cp38-linux_aarch64.whl


```

### TorchVision Install 

- Torchvision is a collection of frequent used datasets, architectures and image algorithms. The installation is simple when you use one of our wheels found on GitHub. Torchvision assumes PyTorch is installed on your machine on the forehand.

- Only for a Jetson Nano with Ubuntu 20.04

- Used with PyTorch 1.12.0

```python

# the dependencies
$ sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
$ sudo pip3 install -U pillow
# install gdown to download from Google drive, if not done yet
$ sudo -H pip3 install gdown
# download TorchVision 0.13.0
$ gdown https://drive.google.com/uc?id=11DPKcWzLjZa5kRXRodRJ3t9md0EMydhj
# install TorchVision 0.13.0
$ sudo -H pip3 install torchvision-0.13.0a0+da3794e-cp38-cp38-linux_aarch64.whl
# clean up
$ rm torchvision-0.13.0a0+da3794e-cp38-cp38-linux_aarch64.whl

```


## **Install  ROS2 foxy**

- please check official website for ros2 foxy here [Link  Here](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)

### step 1 : Set Locale 

- Make sure you have a locale which supports UTF-8. If you are in a minimal environment (such as a docker container), the locale may be something minimal like POSIX. We test with the following settings. However, it should be fine if you’re using a different UTF-8 supported locale.

```python
locale  # check for UTF-8

sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

```

### Step 2: Setup Source 

- Add ROS2 repo to your system 

```python

sudo apt install software-properties-common
sudo add-apt-repository universe

```

- Now add the ROS 2 GPG key with apt.


```python
sudo apt update && sudo apt install curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

```

- Add repo to your source list 

```python

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```


### Step 3 : Install ROS 2 packages

- Update and upgrade  your apt repository caches after setting up the repositories

```python
sudo apt update
sudo apt upgrade
```

- Desktop Install (Recommended): ROS, RViz, demos, tutorials.

```python
sudo apt install ros-foxy-desktop python3-argcomplete
```

- ROS-Base Install (Bare Bones): Communication libraries, message packages, command line tools. No GUI tools.


```python
sudo apt install ros-foxy-ros-base python3-argcomplete
```

- Development tools: Compilers and other tools to build ROS packages

```python
sudo apt install ros-dev-tools
```


### Step 4 Environment setup

-Set up your environment by sourcing the following file.

```python
# Replace ".bash" with your shell if you're not using bash
# Possible values are: setup.bash, setup.sh, setup.zsh
source /opt/ros/foxy/setup.bash
```

- Try following cmd 

```python

ros2 topic list 
```

Output should look like this :- 

```python 

/parameter_events
/rosout

```



### Now create the ROS workspace you are good to go 


#### Optional : After the ROS workplace  source the setup bash for ros and in bashrc file 

```python 
sudo vi .bashrc 
```

And add following lines at the end

```python

source /opt/ros/foxy/setup.bash
source ~/<path to your ros2 ws>/install/setup.bash
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash

```
