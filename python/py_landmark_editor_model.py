
import sys
import xml.etree.ElementTree as ET

import numpy as np
from skimage import transform as tf
#  for transform api

from PIL import Image, ImageTk
#  for jpg support
#  required install python-pil.imagetk

from scipy.spatial.distance import pdist, squareform

from xml.dom import minidom

import os.path
#  for checking file if exist

###########################################
#  The Enumeration class:
#    using the enum define landmark status
###########################################
class Enumeration(object):
    def __init__(self, names):  # or *names, with no .split()
        for number, name in enumerate(names.split()):
            setattr(self, name, number)
##########################################

#  declare the status of landmark
#  for the application
#  eg. LandmarkStatus.VIEW will be 0
LandmarkStatus = Enumeration("VIEW ADDED DELETED")

#########################################
#  The ShapeObject class:
#    using to store shape include image path
#  , box of face, landmarks of face annotation
#########################################
class ShapeObject(object):

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
        return ( int(self._box.get('left'))
            + int(self._box.get('width')) )

    def get_lower(self):
        return ( int(self._box.get('top'))
            + int(self._box.get('height')) )

    def addlandmark(self, landmark, status):
        landmark_dict = {'x': landmark.get('x'),
            'y': landmark.get('y'),
            'status': status}
        self._landmarks.append(landmark_dict)

    def edit_landmark_status(self, landmark_idx, status):
        self._landmarks[landmark_idx]['status'] = status

    def get_normalized_landmarks(self, width = 1.0, height = 1.0):
        normalized_landmarks = []
        self._x_scale = (width
            / float(self._box.get('width')) )
        self._y_scale = (height
            / float(self._box.get('height')) )


        for landmark in self._landmarks:
            #  calculate the new position in normalized width, height
            x_position = (( float(landmark.get('x'))
                - float(self._box.get('left')) )
                * self._x_scale)
            y_position = (( float(landmark.get('y'))
                - float(self._box.get('top')) )
                * self._y_scale)

            #  the status not change because normalize landmark
            #  just change the position of landmark
            landmark_status = landmark.get('status')

            #  push new position into landmark list
            normalized_landmarks.append([
                x_position,
                y_position,
                landmark_status])
        return normalized_landmarks

    def set_tform(self, tform):
        self._tform = tform

    def get_tform(self):
        return self._tform

    def __str__(self):
        #  method for print
        str_to_print = "\nshape_image_path: {}\n".format(self._image_path)
        str_to_print += "\n    shape_box:\
            top:{}, left:{}, width:{}, height:{}\n".format(
            self._box.get('top'),
            self._box.get('left'),
            self._box.get('width'),
            self._box.get('height'))
        str_to_print +=  "\n    shape_landmark({}/{}):\
            x:{}, y:{}\n".format(0,
            len(self._landmarks),
            self._landmarks[0].get('x'),
            self._landmarks[0].get('y'))
        return str_to_print
#  shape_object

class Model():

    def __init__(self, xml_path, width, height):
        self.xml_path = xml_path
        self.shapes = self.parse_xml(xml_path)
        self.width = width
        self.height = height
        self.mean_shape = self.calculate_mean_shape()

        for shape in self.shapes:
            #  print this application current work status
            sys.stdout.write(
                "\rcalculating transform matrix: {}/{}".format(
                self.shapes.index(shape) + 1, len(self.shapes)) )

            tform = self.calcuate_transform_from_shape_to_mean(shape)
            shape.set_tform(tform)
        sys.stdout.write("\nmodel initial finish ! ! !\n")

    def get_shapes(self):
        return self.shapes

    def get_mean_shape(self):
        return self.mean_shape

    #################################################
    #  parse xml and create shape object
    #################################################
    def parse_xml(self, xml_path):
        ##  parse the xml file to full shape_object
        ##  which the xml have ground truth each shape

        #  open the xml for read
        tree = ET.parse(xml_path)
        root = tree.getroot()
        shapes = []

        xml_path = os.path.dirname(os.path.realpath(xml_path))
        sys.stdout.write("\nthe xml dirname:{}\n".format(xml_path))

        #  reading the xml and processing string each line in xml
        for image in root.iter('image'):
            for box in image.iter('box'):
                image_path = ''
                #  add checking image_path valid or not
                if os.path.isfile(image.get('file')):
                    image_path = image.get('file')
                elif os.path.isfile(xml_path + "/" + image.get('file')):
                    image_path = xml_path + "/" + image.get('file')
                else:
                    sys.stdout.write(
                        "\ncan't not found image_path:{}\n".format(
                        image.get('file')) )

                shape = ShapeObject(image_path, box)
                for part in box.iter('part'):
                    shape.addlandmark(part, LandmarkStatus.VIEW)
                shapes.append(shape)
        return shapes

    #  calculate the mean shape of all shape in xml
    #  Before the calculate the mean shape, we normalized all shape in
    #  fixed size square. and get the landmark position in the square

    ###############################################
    #  using normalized location calculate mean
    #  shape's lanmark potion
    ###############################################
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
                self.shapes.index(shape) + 1, len(self.shapes)) )

            nor_landmark = (
                shape.get_normalized_landmarks(self.width, self.height))

            #  push all normalized position in list named nor_landmarks
            #  we will using that calculate average position
            for nor_landmark_part in nor_landmark:
                nor_landmarks.append(nor_landmark_part)

        sys.stdout.write("\n")

        #  calculate the mean value of each landmark
        #  and store to mean shape
        for i in range(len(nor_landmarks) / len(self.shapes)):
            temp_x = 0
            temp_y = 0

            #  calculate how much landmark per shape
            landmark_per_shape = len(nor_landmarks) / len(self.shapes)

            #  calculate the average position each landmark
            for ptr in range(len(self.shapes)):
                temp_x += nor_landmarks[i + landmark_per_shape * ptr][0]
                temp_y += nor_landmarks[i + landmark_per_shape * ptr][1]
            temp_x = temp_x / len(self.shapes)
            temp_y = temp_y / len(self.shapes)

            #  convert list to dict
            temp_landmark = {'x': temp_x, 'y': temp_y}

            #  push the average landmark into mean_shape
            mean_shape.addlandmark(temp_landmark, LandmarkStatus.VIEW)

        return mean_shape

    ######################################################
    #  estimate the landmark position in shape
    #  space from mean space position and transform matrix
    ######################################################
    def estimate_shape_from_mean(self, shape):

        #  the landmark we estimate keep the landmark status
        #  from mean shape, this will store their status
        landmark_status = []

        for landmark_part in self.mean_shape.get_normalized_landmarks(
                self.width, self.height):
            #  copy the landmark status form mean_shape
            landmark_status.append(landmark_part[2])

        #  getting the shape transform matrix
        #  from shape , the we can accroding that
        #  transform mean space to shape space
        tform = shape.get_tform()

        src = self.convert_landmarks_to_nparray(
            self.mean_shape.get_normalized_landmarks(self.width, self.height))

        #  inverse transform from mean space to sample space
        dst = tform.inverse(src)

        #  the dst_shape store shape calculate from mean shape
        #  trans the dst from array to list
        dst_shape = dst.tolist()

        #  declare rhe estimate_shape store shape we estimate from
        #  mean shape with tform,
        #  init ShapeObject using image_path and box equal to origin
        estimate_shape = ShapeObject(shape._image_path, shape._box)

        #  un-normalize position of position to fit origin box
        #  re-warp the list to dict object
        for i in range(len(dst)):
            #  un-normalize landmark position
            temp_x = (dst_shape[i][0]
                / shape._x_scale
                + float(shape._box.get('left')) )
            temp_y = (dst_shape[i][1]
                / shape._y_scale
                + float(shape._box.get('top')) )


            #  warp to dict
            estimate_landmark = {'x': temp_x, 'y': temp_y}
            #  push into estimate_shape
            estimate_shape.addlandmark(
                estimate_landmark,
                landmark_status[i])

        return estimate_shape

    ##################################################
    #  this function provid us convert the landmark
    #  format (in shape) to numpy.arrary
    ##################################################
    def convert_landmarks_to_nparray(self, landmarks):
        #  convert the landmark position from
        #  list to nparray
        points = []

        for landmark_part in landmarks:
            points.append(landmark_part[0])
            points.append(landmark_part[1])

        src = np.array(points).reshape(
            (len(points) / 2, 2))
        return src

    #########################################################
    #  calculate the transform from shape (in xml) to
    #  mean shape (we calculate), accroding this we can
    #  estimate new landmark point from  mean space to
    #  each shape space
    #########################################################
    def calcuate_transform_from_shape_to_mean(self, shape):
        #  calculate the transform between shapes (ground trurh and mean)

        #  we convert the landmark format to numpy.array
        #  src store the landmark position information of shape in xml
        #  dst store the landmark position information of mean shape
        src = self.convert_landmarks_to_nparray(
            shape.get_normalized_landmarks(self.width, self.height))

        dst = self.convert_landmarks_to_nparray(
            self.mean_shape.get_normalized_landmarks(self.width, self.height))

        #  after format converted we can using numpy library find out
        #  transform matrix
        tform = tf.estimate_transform(
            'piecewise-affine', src, dst)

        #  print "\nthe tform form normalized shape to mean shape:\n"
        #  print tform._matrix
        return tform

    #############################################################
    #  calculate the mean shape image, of all image in xml
    #  with transform, then we can display the image and mean
    #  shape together
    #############################################################
    def calculate_mean_shape_image(self):
        #  create the image_arry shape in width x height x channels
        #  to store the average pixel of all image

        #  checking the Average.png if exist
        if os.path.isfile("Average.png"):
            sys.stdout.write("\nAverage.png existed, skip re-calculate ! ! !\n")
            self.mean_shape._image_path = "Average.png"
            return

        image_arry = np.zeros(
            (self.width, self.height, 3),
            dtype=np.float)

        for shape in self.shapes:
            #  print the current work status
            sys.stdout.write( "\rcalculating mean image: {}/{} ".format(
                self.shapes.index(shape) + 1 ,
                len(self.shapes)) )

                    #  open image from path and convert to RGB
            image = Image.open(shape.get_image_path()).convert('RGB')

            #  crop the image by box
            image = image.crop(
                box=(shape.get_left(),
                shape.get_upper(),
                shape.get_right(),
                shape.get_lower()) )
            image = image.resize(
                size=(int(self.width), int(self.height)) )

            #  warping image with tform
            #  first: convert the image to array
            imarr = np.array(
                image.getdata(),
                dtype=np.float).reshape(image.size[0], image.size[1], 3)


            #  second: getting tform between shape
            tform = shape.get_tform()

            #  third: warp image
            warped = tf.warp(
                imarr, tform,
                output_shape=(self.width, self.height))

            warped_image = imarr * warped

            image_arry = image_arry + warped_image / len(self.shapes)

        image_arry = np.array(np.round(image_arry), dtype=np.uint8)

        out=Image.fromarray(image_arry, mode=None)

        sys.stdout.write("\nstore Averge.png at CWD\n")
        out.save("Average.png")
        self.mean_shape._image_path = "Average.png"

    #######################################################
    #  direct add point x,y as new landmark into mean shape
    #  with landmark status: ADDED
    #######################################################
    def add_landmark_to_mean_shape(self, x, y):
        landmark = {'x': x, 'y': y}
        self.mean_shape.addlandmark(landmark, LandmarkStatus.ADDED)

    #######################################################
    #  find out the nearest landmark from x,y in mean shape
    #######################################################
    def compute_nearest_landmark(self, x, y):
        #  we want to using numpy library to calculate distance
        #  betweent, need to convert landmark to numpy.array format
        compare_to = self.convert_landmarks_to_nparray(
            self.mean_shape.get_normalized_landmarks(self.width, self.height))
        compare_from = np.array([x, y])

        #  push x,y to last row of numpy.array
        compare = np.vstack((compare_to, compare_from))

        #  calculate the point-to-point distance
        #  and take out we need: point(x,y) to other points
        compare_result = squareform(pdist(compare, 'euclidean'))[:-1,-1]

        #  get which landmark has minimum distance
        nearest_idx = np.argmin(compare_result)

        sys.stdout.write("\nnearest is {}-th landmark\n".format(nearest_idx))

        return nearest_idx

    #######################################################
    #  remove the nearest landmark from x,y in mean shape
    #######################################################
    def remove_landmark_from_mean_shape(self, x, y):
        #  we want to remove the nearest landmark, first we
        #  find out which landmark is nearest
        target_landmark_idx = self.compute_nearest_landmark(x, y)

        #  change landmark status to DELETED to target landmark
        self.mean_shape.edit_landmark_status(
            target_landmark_idx,
            LandmarkStatus.DELETED)

        sys.stdout.write("\nremoved {}-th landmark\n".format(
            target_landmark_idx))

    #########################################################
    #  Building the xml document form shapes with landmark
    #  , landmark status
    #########################################################
    def write_xml_to_file(self, filename):
        # prepare shapes for write
        shapes = []
        for shape in self.shapes:
            shapes.append(self.estimate_shape_from_mean(shape))

        # Add desired xml declare and processing instructions.
        xml_head = "<?xml version='{}' encoding='{}'?>\n".format(
            "1.0",
            "ISO-8859-1")
        xml_pi = "<?xml-stylesheet type='{}' href='{}'?>\n".format(
            "text/xsl",
            "image_metadata_stylesheet.xsl")

        #  open the file to write
        target_file = open(filename, 'w')

        #  write head to xml file
        target_file.write(xml_head)
        target_file.write(xml_pi)

        target_file.write("<dataset>\n")
        target_file.write("<name>imglab dataset</name>\n")
        target_file.write(
            "<comment>Created by {}.</comment>\n".format(__name__))
        target_file.write("<images>\n")

        #  write shapes to xml file
        for shape in shapes:
            target_file.write(
                "  <image file='{}'>\n".format(shape.get_image_path()))
            target_file.write("     <box")
            target_file.write(" top = '{}'".format(shape._box.get('top')))
            target_file.write(" left = '{}'".format(shape._box.get('left')))
            target_file.write(" width = '{}'".format(shape._box.get('width')))
            target_file.write(" height = '{}'>\n"
                .format(shape._box.get('height')))

            count = 0
            for landmark in shape._landmarks:
                if landmark.get('status')!=LandmarkStatus.DELETED:
                    target_file.write(
                        "         <part name='{}' x='{}' y='{}'/>\n".format(
                        count,
                        int(landmark.get('x')),
                        int(landmark.get('y'))) )
                    count = count + 1

            target_file.write("     </box>\n")
            target_file.write("  </image>\n")

        #  write foot to xml file
        target_file.write("</images>\n")
        target_file.write("</dataset>")

        target_file.close()
        sys.stdout.write("\nwrite xml:{} finished ! ! ! \n".format(filename))
