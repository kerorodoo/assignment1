
import sys
import xml.etree.ElementTree as ET
import Tkinter

xml_path = sys.argv[1]
shapes = []
WIDTH = 200.0
HEIGHT = 200.0
CIRCLE_R = 2
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

##  parse the xml file to full shape_object
##  which the xml have ground truth each shape
#  open the xml for read
tree = ET.parse(xml_path)
root = tree.getroot()

#  reading the xml and processing string each line in xml
for image in root.iter('image'):
    for box in image.iter('box'):
        image_path = image.get('file');
        shape = ShapeObject(image_path, box)
        for part in box.iter('part'):
            shape.addlandmark(part)
        shapes.append(shape)

#  calc the mean shape of all shape in xml
#  Before the calc the mean shape, we normalized all shape in
#  fixed size square. and get the landmark position in the square

#  we declare nor_landmarks variable to store all landmark position
#  after normalized.
#  and mean_landmarks variable store the mean shape.
nor_landmarks = []
mean_landmarks = []

for shape in shapes:
    nor_landmark = shape.get_normalized_landmarks(WIDTH, HEIGHT)
    for nor_landmark_part in nor_landmark:
        nor_landmarks.append(nor_landmark_part)

print "\nnor landmarks len is {}\n".format(len(nor_landmarks))
print "\nshapes len is {}\n".format(len(shapes))
print "\nper shape contain {} landmarks\n".format(len(nor_landmarks)/len(shapes))

#  calc the mean value of each landmark
#  and store to mean shape
for i in range(len(nor_landmarks)/len(shapes)):
    temp_x = 0
    temp_y = 0
    landmark_per_shape = len(nor_landmarks)/len(shapes)
    for ptr in range(len(shapes)):
        temp_x += nor_landmarks[i + landmark_per_shape * ptr][0]
        temp_y += nor_landmarks[i + landmark_per_shape * ptr][1]
    temp_x = temp_x / len(shapes)
    temp_y = temp_y / len(shapes)
    mean_landmarks.append([temp_x, temp_y])


#  display the mean shape in gui window
window_width = int(WIDTH)
window_height = int(HEIGHT)
window = Tkinter.Tk()
canvas = Tkinter.Canvas(window, width=window_width, height=window_height, bg="#000000")
canvas.pack()
img = Tkinter.PhotoImage(width=window_width, height=window_height)
canvas.create_image((window_width/2, window_height/2), image=img, state="normal")
for part in mean_landmarks:
    circle_left = int(part[0]) - CIRCLE_R/2
    circle_top = int(part[1]) - CIRCLE_R/2
    circle_right = int(part[0]) + CIRCLE_R/2
    circle_buttom = int(part[1]) + CIRCLE_R/2
    canvas.create_oval(circle_left, circle_top, circle_right, circle_buttom, outline="red", fill="green", width=2)
window.mainloop()
#  display the landmark transform mean shape, ground truth and image in new
#  window when finger out the image

# Code to add widgets will go here...

