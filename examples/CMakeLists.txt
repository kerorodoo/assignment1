#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#


cmake_minimum_required(VERSION 2.8.4)

PROJECT(examples)

include(../dlib/cmake)

# Tell CMake to compile a program.  We do this with the ADD_EXECUTABLE()
# statement which takes the name of the output executable and then a list of
# .cpp files to compile.  Here each example consists of only one .cpp file but
# in general you will make programs that const of many .cpp files.
#ADD_EXECUTABLE(assignment_learning_ex assignment_learning_ex.cpp)
# Then we tell it to link with dlib.
#TARGET_LINK_LIBRARIES(assignment_learning_ex dlib)

#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# Since there are a lot of examples I'm going to use a macro to simply this
# CMakeLists.txt file.  However, usually you will create only one executable in
# your cmake projects and use the syntax shown above.
MACRO(add_example name)
   ADD_EXECUTABLE(${name} ${name}.cpp)
   TARGET_LINK_LIBRARIES(${name} dlib )
ENDMACRO()

# The deep learning toolkit requires a C++11 capable compiler.  
#if (COMPILER_CAN_DO_CPP_11)
#   add_example(dnn_introduction_ex)
#   add_example(dnn_introduction2_ex)
#   add_example(dnn_inception_ex)
#   add_example(dnn_imagenet_ex)
#   add_example(dnn_imagenet_train_ex)
#endif()

#here we apply our macros
add_example(face_detection_ex)
add_example(face_landmark_detection_adaptive)
add_example(face_landmark_detection_ex)
add_example(face_landmark_detection_noline)
add_example(train_shape_predictor_adaptive)
add_example(xml_parser_mine)




if (DLIB_NO_GUI_SUPPORT)
   message("No GUI support, so we won't build the webcam_face_pose_ex example.")
else()
   find_package(OpenCV QUIET)
   set(OpenCV_FOUND "/usr/local")
   if (OpenCV_FOUND)
      
       include_directories(${OpenCV_INCLUDE_DIRS})

       ADD_EXECUTABLE(webcam_face_pose_ex webcam_face_pose_ex.cpp)
       TARGET_LINK_LIBRARIES(webcam_face_pose_ex dlib ${OpenCV_LIBS} )
       
       ADD_EXECUTABLE(webcam_face_pose_with_chip_detail webcam_face_pose_with_chip_detail.cpp)
       TARGET_LINK_LIBRARIES(webcam_face_pose_with_chip_detail dlib ${OpenCV_LIBS})

       ADD_EXECUTABLE(antispoof_face_lbf_svm_with_svd antispoof_face_lbf_svm_with_svd.cpp)
       TARGET_LINK_LIBRARIES(antispoof_face_lbf_svm_with_svd dlib ${OpenCV_LIBS})
   
       ADD_EXECUTABLE(main main.cpp)
       TARGET_LINK_LIBRARIES(main dlib ${OpenCV_LIBS})
   else()
      message("OpenCV not found, so we won't build the webcam_face_pose_ex example.")
   endif()
endif()

#if (DLIB_LINK_WITH_SQLITE3)
#   add_example(sqlite_ex)
#endif()


