# assignment1
## compile c++ example:
```
    cd examples
    mkdir build
    cmake ..
    cmake --build .
```
## compile python-example (if want to using dlib python interface):
```
    apt-get install setuptools
    apt-get install boost-python-dev
    python setup.py install
```
> setup.py provid from dlib [DLIB Github](https://github.com/davisking/dlib)

## using python application:
#### 1. python/py_trans_pts2xml.py
###### this application is using for create dataset(.xml) from .pts file (which be containd in [dlib provid dataset:ibug_300W_large_face_landmark_dataset] (http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz))
###### usage:
```
    >python py_trans_pts2xml.py [root_folder_of_dataset] [number_of_samples_in_output_xml_file]
```
#### 2. python/py_landmark_editor.py and py_landmark_editor_model.py
######  this application is using for create dataset(.xml) from exist dataset. After edit landmark annotation by this application we can export it to new xml as dataset for dlib.
######  usage:
```
    >python py_landmark_editor.py [path_of_xml_file]
```
```
    right-mouse-click: remove nearest landmark from mean shape
    left-mouse-click: add landmark from mean shape
    keyboard-'s': export to xml (test.xml)
    keyboard-'r': re-calculate mean shape image 
```
## using c++ applications:
#### 1. examples/train_shape_predictor_adaptive.cpp

