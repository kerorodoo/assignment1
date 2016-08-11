
import sys
import xml.etree.ElementTree as ET

import numpy as np
from skimage import transform as tf

#  shape_object
class ShapeObject(object):

    def __init__(self, image_path, box, landmarks):
        self._image_path = image_path
        self._box = box
        self._landmarks = landmarks

    def __init__(self, image_path, box):
        self._image_path = image_path
        self._box = box
        self._landmarks = []

    def get_left(self):
        return int(self._box.get('left'))

    def get_upper(self):
        return int(self._box.get('top'))

    def get_right(self):
        return (int(self._box.get('left')) + int(self._box.get('width')))

    def get_lower(self):
        return (int(self._box.get('top')) + int(self._box.get('height')))

    def addlandmark(self, landmark):
        self._landmarks.append(landmark)

    def get_normalized_landmarks(self, width = 1.0, height = 1.0):
        normalized_landmarks = []
        x_scale = width / float(self._box.get('width'))
        y_scale = height / float(self._box.get('height'))
        for landmark in self._landmarks:
            x_position = (float(landmark.get('x')) - float(self._box.get('left'))) * x_scale
            y_position = (float(landmark.get('y')) - float(self._box.get('top'))) * y_scale
            normalized_landmarks.append([x_position, y_position])
        return normalized_landmarks

    def __str__(self):
        #  method for print
        str_to_print = "\nshape_image_path: {}\n".format(self._image_path)
        str_to_print += "\n    shape_box: top:{}, left:{}, width:{}, height:{}\n".format(self._box.get('top'), self._box.get('left'), self._box.get('width'), self._box.get('height'))
        str_to_print +=  "\n    shape_landmark({}/{}): x:{}, y:{}\n".format(1, len(self._landmarks), self._landmarks[0].get('x'), self._landmarks[0].get('y'))
        return str_to_print
#  shape_object

class Model():

    def __init__(self, xml_path, width, height):
        self.xml_path = xml_path
        self.shapes = self.parse_xml(xml_path)
        self.width = width
        self.height = height
        self.mean_landmarks = self.calculate_mean_landmarks()

    def get_shapes(self):
        return self.shapes

    def get_mean_shape(self):
        return self.mean_landmarks

    def parse_xml(self, xml_path):
        ##  parse the xml file to full shape_object
        ##  which the xml have ground truth each shape

        #  open the xml for read
        tree = ET.parse(xml_path)
        root = tree.getroot()
        shapes = []

        #  reading the xml and processing string each line in xml
        for image in root.iter('image'):
            for box in image.iter('box'):
               image_path = image.get('file');
               shape = ShapeObject(image_path, box)
               for part in box.iter('part'):
                   shape.addlandmark(part)
               shapes.append(shape)
        return shapes

    #  calculate the mean shape of all shape in xml
    #  Before the calculate the mean shape, we normalized all shape in
    #  fixed size square. and get the landmark position in the square

    def calculate_mean_landmarks(self):
        #  we declare nor_landmarks variable to store all landmark position
        #  after normalized.
        #  and mean_landmarks variable store the mean shape.
        nor_landmarks = []
        mean_landmarks = []

        for shape in self.shapes:
            nor_landmark = shape.get_normalized_landmarks(self.width, self.height)
            for nor_landmark_part in nor_landmark:
                nor_landmarks.append(nor_landmark_part)

        #print "\nnor landmarks len is {}\n".format(len(nor_landmarks))
        #print "\nshapes len is {}\n".format(len(self.shapes))
        #print "\nper shape contain {} landmarks\n".format(len(nor_landmarks)/len(self.shapes))

        #  calculate the mean value of each landmark
        #  and store to mean shape
        for i in range(len(nor_landmarks)/len(self.shapes)):
            temp_x = 0
            temp_y = 0
            landmark_per_shape = len(nor_landmarks)/len(self.shapes)
            for ptr in range(len(self.shapes)):
                temp_x += nor_landmarks[i + landmark_per_shape * ptr][0]
                temp_y += nor_landmarks[i + landmark_per_shape * ptr][1]
            temp_x = temp_x / len(self.shapes)
            temp_y = temp_y / len(self.shapes)
            mean_landmarks.append([temp_x, temp_y])
        return mean_landmarks

    def calcuate_transform_form_shape_to_mean(self, shape_ptr):
        #  calculate the transform between shapes (ground trurh and mean)
        src_points = []
        dst_points = []
        for landmark_part in shapes[shape_ptr].get_normalized_landmarks(self.width, self.height):
            src_points.append(landmark_part[0])
            src_points.append(landmark_part[1])
        for landmark_part in mean_landmarks:
            dst_points.append(landmark_part[0])
            dst_points.append(landmark_part[1])

        src = np.array(src_points).reshape((len(src_points)/2, 2))
        dst = np.array([dst_points]).reshape((len(dst_points)/2, 2))

        tform = tf.estimate_transform('similarity', src, dst)

        print "\nthe tform form normalized shape to mean shape:\n"
        print tform._matrix
        return tform
