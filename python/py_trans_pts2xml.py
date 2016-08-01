import sys
import os
import glob
import random

pts_folder_path = sys.argv[1]
pts_file_count = 0
pts_all = []
class PTS_Object(object):
    def __init__(self, _file_path, _landmark_info, _landmarks):
        self._file_path = _file_path
        self._image_path = _file_path.replace(".pts", ".jpg")
        if _file_path.find("lfpw") > 0:
            self._image_path = _file_path.replace(".pts", ".png")
        self._landmark_info = _landmark_info
        self._landmarks = _landmarks
        self._box_left = 0
        self._box_top = 0
        self._box_width = 388
        self._box_height = 388
    def adjust_box(self):
        x = []
        y = []
        for landmark_ptr in self._landmarks:
            x.append(float(landmark_ptr[0]))
            y.append(float(landmark_ptr[1]))
        self._box_left = min(x)
        self._box_top = min(y)
        self._box_width = max(x) - min(x)
        self._box_height = max(y) - min(y)
    def adjust_precision(self):
        self._box_left = int(self._box_left)
        self._box_top = int(self._box_top)
        self._box_width = int(self._box_width)
        self._box_height = int(self._box_height)
        for _ptr in range(len(self._landmarks)):
            self._landmarks[_ptr][0] = int(float(self._landmarks[_ptr][0]))
            self._landmarks[_ptr][1] = int(float(self._landmarks[_ptr][1]))

#full pts_object
pts_in_folder = glob.glob(os.path.join(pts_folder_path, "*.pts"))
pts_in_folder += glob.glob(os.path.join(pts_folder_path+'/*/', "*.pts"))
pts_in_folder += glob.glob(os.path.join(pts_folder_path+'/*/*/', "*.pts"))
for file_path in pts_in_folder:
    pts_file_count = pts_file_count + 1
    print("Processing file({}/{}): {}".format(pts_file_count, len(pts_in_folder), file_path))
    file_ptr = open(file_path, 'r')

    landmark_info = 0
    landmarks = []

    for line in file_ptr.readlines():
        line = line.replace('\n','')
        if ('n_points:' in line):
            landmark_info = line.rsplit('  ')[1]
        if (' ' in line) and not ('n_points:' in line) and not ('{' in line) and not ('}' in line) and not ('version:' in line):
            landmarks.append(line.rsplit(' '))
    pts_all.append(PTS_Object(file_path, landmark_info, landmarks))
#full pts_object

#adjust_box
for pts in pts_all:
    pts.adjust_box()
#adjust_box
#adjust_precision
for pts in pts_all:
    pts.adjust_precision()
#adjust_precision

#function create_xml
def create_xml(target_file_name, pts_selected):
    #write xml
    xml_head = []
    xml_head.append("<?xml version='1.0' encoding='ISO-8859-1'?>")
    xml_head.append("<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>")
    xml_head.append("<dataset>")
    xml_head.append("<name>imglab dataset</name>")
    xml_head.append("<comment>Created by {}.</comment>".format(sys.argv[0]))
    xml_head.append("<images>")
    xml_body = []
    xml_body.append("  <image file='{}'>")
    xml_body.append("    <box top = '{}' left = '{}' width = '{}' height = '{}'>")
    xml_body.append("      <part name='{}' x='{}' y='{}'/>")
    xml_body.append("    </box>")
    xml_body.append("  </image>")
    xml_foot = []
    xml_foot.append("</images>")
    xml_foot.append("</dataset>")

    print("\nwe are going to create xml: {}\n".format(target_file_name))
    target_file_ptr = open(target_file_name, 'w')
    for line in xml_head:
        target_file_ptr.write(line + "\n")
    for pts in pts_selected:
        target_file_ptr.write(xml_body[0].format(pts._image_path) + "\n")
        target_file_ptr.write(xml_body[1].format(pts._box_top, pts._box_left, pts._box_width, pts._box_height) + "\n")
        for landmark_selected in range(len(pts._landmarks)):
            target_file_ptr.write(xml_body[2].format(landmark_selected, pts._landmarks[landmark_selected][0], pts._landmarks[landmark_selected][1]) + "\n")
        target_file_ptr.write(xml_body[3] + "\n")
        target_file_ptr.write(xml_body[4] + "\n")
    for line in xml_foot:
        target_file_ptr.write(line + "\n")
    target_file_ptr.close()
    #write xml
#function create xml

#random pick pts
#to avoid the dupicate selection record selected in pts_cant_select
pts_cant_select = []
if (len(sys.argv)) == 2:
    # the case for not provide how much picture in the xml
    # that will create output.xml containing all picture
    target_file_name = "output.xml"
    create_xml(target_file_name, pts_all)
if (len(sys.argv)) > 2:
    # the input error detection
    sample_need = 0
    for argument_ptr in range(2, len(sys.argv)):
        sample_need = sample_need + int(sys.argv[argument_ptr])
    if sample_need > len(pts_all):
        print("\nthe image not enought    !!!!!\n")
        quit()
if (len(sys.argv)) > 2:
    # the case for already provide program how much picture in xml
    # each number will create sparate xml file contain the number of picture
    for argument_ptr in range(2, len(sys.argv)):
        target_file_name = ("output{}.xml".format(sys.argv[argument_ptr]))
        pts_selected = []
        for pts_ptr in range(int(sys.argv[argument_ptr])):
            # random select object and avoid duplicate selection
            select = random.randint(0, len(pts_all) - 1)
            while select in pts_cant_select:
                select = random.randint(0, len(pts_all) - 1)
            # record the selection
            pts_cant_select.append(select)
            pts_selected.append(pts_all[select])
        create_xml(target_file_name, pts_selected)
#random pick pts
