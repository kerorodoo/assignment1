
import sys
import xml.etree.ElementTree as ET

import numpy as np
from skimage import transform as tf
#  for transform api

from PIL import Image, ImageTk
#  for jpg support
#  required install python-pil.imagetk

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

    def get_image_path(self):
        return self._image_path

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
        self._x_scale = width / float(self._box.get('width'))
        self._y_scale = height / float(self._box.get('height'))


        for landmark in self._landmarks:
            x_position = (float(landmark.get('x')) - float(self._box.get('left'))) * self._x_scale
            y_position = (float(landmark.get('y')) - float(self._box.get('top'))) * self._y_scale
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
        self.mean_shape = self.calculate_mean_shape()

    def get_shapes(self):
        return self.shapes

    def get_mean_shape(self):
        return self.mean_shape

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

    def calculate_mean_shape(self):
        #  we declare nor_landmarks variable to store all landmark position
        #  after normalized.
        #  and mean_landmarks variable store the mean shape.
        nor_landmarks = []

        box = {'left': 0, 'top': 0, 'width': self.width, 'height': self.height}

        mean_shape = ShapeObject('', box)

        for shape in self.shapes:
            #  print the program working staus
            sys.stdout.write(
                    "\rcalculating normalized landmark location: {}/{}".format(
                        self.shapes.index(shape) + 1, len(self.shapes)))

            nor_landmark = (
                    shape.get_normalized_landmarks(self.width, self.height))

            for nor_landmark_part in nor_landmark:
                nor_landmarks.append(nor_landmark_part)

        sys.stdout.write("\n")

        #  calculate the mean value of each landmark
        #  and store to mean shape
        for i in range(len(nor_landmarks) / len(self.shapes)):
            temp_x = 0
            temp_y = 0
            landmark_per_shape = len(nor_landmarks) / len(self.shapes)

            for ptr in range(len(self.shapes)):
                temp_x += nor_landmarks[i + landmark_per_shape * ptr][0]
                temp_y += nor_landmarks[i + landmark_per_shape * ptr][1]
            temp_x = temp_x / len(self.shapes)
            temp_y = temp_y / len(self.shapes)

            temp_landmark = {'x': temp_x, 'y': temp_y}
            mean_shape.addlandmark(temp_landmark)

        return mean_shape

    def calcuate_shape_from_mean(self, shape):
        #  calculate the shape from mean shape
        tform = self.calcuate_transform_from_shape_to_mean(shape)

        src_points = []

        for landmark_part in self.mean_shape.get_normalized_landmarks(self.width, self.height):
            src_points.append(landmark_part[0])
            src_points.append(landmark_part[1])

        src = np.array(src_points).reshape((len(src_points) / 2, 2))

        #  inverse transform from mean space to sample space
        dst = tform.inverse(src)

        #  the dst_shape store shape calculate from mean shape
        #  trans the dst from array to list
        dst_shape = dst.tolist()

        #  create the ShapeObject to store the shape we calculate
        estimate_shape = ShapeObject(shape._image_path, shape._box)

        #  re-warp the list to dict object
        for i in range(len(dst)):
            temp_x = dst_shape[i][0] / shape._x_scale + (
                    float(shape._box.get('left')))
            temp_y = dst_shape[i][1] / shape._y_scale + (
                    float(shape._box.get('top')))

            estimate_landmark = {'x': temp_x, 'y': temp_y}
            estimate_shape.addlandmark(estimate_landmark)

        return estimate_shape


    def calcuate_transform_from_shape_to_mean(self, shape):
        #  calculate the transform between shapes (ground trurh and mean)
        src_points = []
        dst_points = []
        for landmark_part in shape.get_normalized_landmarks(self.width, self.height):
            src_points.append(landmark_part[0])
            src_points.append(landmark_part[1])

        for landmark_part in self.mean_shape.get_normalized_landmarks(self.width, self.height):
            dst_points.append(landmark_part[0])
            dst_points.append(landmark_part[1])

        src = np.array(src_points).reshape((len(src_points)/2, 2))
        dst = np.array([dst_points]).reshape((len(dst_points)/2, 2))

        tform = tf.estimate_transform('piecewise-affine', src, dst)


        #  print "\nthe tform form normalized shape to mean shape:\n"
        #  print tform._matrix
        return tform

    def calculate_mean_shape_image(self):
        #  create the image_arry shape in width x height x channels
        #  to store the average pixel of all image
        #
        image_arry = np.zeros((self.width, self.height, 3), dtype=np.float)

        for shape in self.shapes:
            #  print the current work status
            sys.stdout.write(
                    "\rcalculating mean image: {}/{} ".format(
                        self.shapes.index(shape) + 1 , len(self.shapes)))

            #  open image from path and convert to RGB
            image = Image.open(shape.get_image_path()).convert('RGB')

            #  crop the image by box
            image = image.crop(box=(shape.get_left(), shape.get_upper(), shape.get_right(), shape.get_lower()))
            image = image.resize(size=(int(self.width), int(self.height)))

            #  warping image with tform
            #  first: convert the image to array
            imarr = np.array(image.getdata(), dtype=np.float).reshape(image.size[0], image.size[1], 3)


            #  second: getting tform between shape
            tform = self.calcuate_transform_from_shape_to_mean(shape)

            #  third: warp image
            warped = tf.warp(imarr, tform, output_shape=(self.width, self.height))
            warped_image = imarr * warped

            image_arry = image_arry + warped_image / len(self.shapes)

        image_arry = np.array(np.round(image_arry), dtype=np.uint8)

        out=Image.fromarray(image_arry, mode=None)

        sys.stdout.write("\nstore Averge.png at CWD\n")
        out.save("Average.png")
        self.mean_shape._image_path = "Average.png"
