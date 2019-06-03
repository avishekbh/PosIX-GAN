# PosIX-GAN
The official repository for "PosIX-GAN: Generating multiple poses using GAN for Pose-Invariant Face Recognition"; GMDL (ECCVW) 2018

The code has been written for python 2.7. <br />
### Requirements: ###

tensorflow-gpu==1.8.0 <br />
keras==2.1.6 <br />
graphviz==0.10.1 <br />
matplotlib==2.2.4 <br />
Pillow==6.0.0 <br />
pydot==1.4.1 <br />
scipy==1.2.1 <br />

Also, make sure to install python-tk: sudo apt-get install python-tk

Please note that the data preparation task is vital and the following steps must be followed:  

* Normalise the image data and save into npy files

* Generate the corresponding labels in one-hot vector fashion.

* Make sure the separate set of the image data has been prepared to evaluate the PMSE loss. There should be nine separate numpy files (containing images) for each of the nine generator outputs.

* Make sure all packages have been installed, of the correct version.

* Make changes to the data loader and train functions accordingly.

After above mentioned changes have been made, to start training, run:

python main.py

Once training has been completed, to test the model performance, run:

python test.py

If you are using this code for your work, please cite:
<br />
<br />
@inproceedings{bhattacharjee2018posix,<br />
  title={PosIX-GAN: Generating multiple poses using GAN for Pose-Invariant Face Recognition},  
  author={Bhattacharjee, Avishek and Banerjee, Samik and Das, Sukhendu},  
  booktitle = {The European Conference on Computer Vision (ECCV) Workshops},  
  month = {September},  
  year = {2018}<br />
}
