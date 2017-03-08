// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    

    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead 
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <string>
#include <chrono>

using namespace dlib;
using namespace std;


//----------------------------------------------------------------------------------------

std::vector<int> get_around_chin_ear_to_ear_part(int num_parts)
{
    std::vector<int> set;
    if(num_parts == 68)
    {
        for (unsigned long i = 0; i <= 16; i++)
            set.push_back(i);
        return set;
    }
    else
    {
       for (unsigned long i = 0; i <= 10; i++)
            set.push_back(i);
       for (unsigned long i = 21; i <= 109; i+=11)
            set.push_back(i);
       for (unsigned long i = 114; i <= 134; i++)
            set.push_back(i);
       return set;
    }
}


std::vector<int> get_top_of_nose(int num_parts)
{
    std::vector<int> set;
    if(num_parts == 68)
    {
        for (unsigned long i = 27; i <= 30; i++)
            set.push_back(i);
        return set;
    }
    else
    {
       return set;
    }
}


std::vector<int> get_left_eyebrow(int num_parts)
{
    std::vector<int> set;
    if(num_parts == 68)
    {
        for (unsigned long i = 18; i <= 21; i++)
            set.push_back(i);
        return set;
    }
    else
    {
        for (unsigned long i = 92; i <= 97; i++)
            set.push_back(i);
        for (unsigned long i = 99; i <= 108; i++)
            set.push_back(i);
        for (unsigned long i = 110; i <= 113; i++)
            set.push_back(i);
        return set;
    }
}

std::vector<int> get_right_eyebrow(int num_parts)
{
    std::vector<int> set;
    if(num_parts == 68)
    {
        for (unsigned long i = 22; i <= 26; i++)
            set.push_back(i);
        return set;
    }
    else
    {
        for (unsigned long i = 70; i <= 75; i++)
            set.push_back(i);
        for (unsigned long i = 77; i <= 86; i++)
            set.push_back(i);
        for (unsigned long i = 88; i <= 91; i++)
            set.push_back(i);
        return set;
    }
}


std::vector<int> get_bottom_of_nose(int num_parts)
{
    std::vector<int> set;
    if(num_parts == 68)
    {
        for (unsigned long i = 30; i <= 35; i++)
            set.push_back(i);
        set.push_back(30);
        return set;
    }
    else
    {
        for (unsigned long i = 135; i <= 151; i++)
            set.push_back(i);
        return set;
    }
}


std::vector<int> get_left_eye(int num_parts)
{
    std::vector<int> set;
    if(num_parts == 68)
    {
        for (unsigned long i = 36; i <= 41; i++)
            set.push_back(i);
        return set;
    }
    else
    {
        for (unsigned long i = 48; i <= 53; i++)
            set.push_back(i);
        for (unsigned long i = 55; i <= 64; i++)
            set.push_back(i);
        for (unsigned long i = 66; i <= 69; i++)
            set.push_back(i);
        return set;
    }

}


std::vector<int> get_right_eye(int num_parts)
{
    std::vector<int> set;
    if(num_parts == 68)
    {
        for (unsigned long i = 42; i <= 47; i++)
            set.push_back(i);
        return set;
    }
    else
    {
        for (unsigned long i = 26; i <= 31; i++)
            set.push_back(i);
        for (unsigned long i = 33; i <= 42; i++)
            set.push_back(i);
        for (unsigned long i = 44; i <= 47; i++)
            set.push_back(i);
        return set;
    }
}


std::vector<int> get_lips_outer_part(int num_parts)
{
    std::vector<int> set;
    if(num_parts == 68)
    {
        for (unsigned long i = 48; i <= 59; i++)
            set.push_back(i);
        return set;
    }
    else
    {
        for (unsigned long i = 152; i <= 179; i++)
            set.push_back(i);
        return set;
    }
}


std::vector<int> get_lips_inside_part(int num_parts)
{
    std::vector<int> set;
    if(num_parts == 68)
    {
        for (unsigned long i = 60; i <= 67; i++)
            set.push_back(i);
        return set;
    }
    else
    {
        for (unsigned long i = 11; i <= 20; i++)
            set.push_back(i);
        for (unsigned long i = 23; i <= 25; i++)
            set.push_back(i);
        for (unsigned long i = 180; i <= 193; i++)
            set.push_back(i);
        return set;
    }
}


std::vector<image_window::overlay_line> render_face_detections_adaptive(
    const std::vector<full_object_detection>& dets,
    const rgb_pixel color = rgb_pixel(0,255,0)
)
{
   std::vector<image_window::overlay_line> lines;
        for (unsigned long i = 0; i < dets.size(); ++i)
        {
           const full_object_detection& d= dets[i];
           // Around Chin. Ear to Ear
           std::vector<int> set = get_around_chin_ear_to_ear_part(dets[i].num_parts());
           for (unsigned long pen = 1; pen < set.size(); pen++)
               lines.push_back(image_window::overlay_line(d.part(set[pen]), d.part(set[pen-1]), color));

           // Line on top of nose
           set = get_top_of_nose(dets[i].num_parts());
           for (unsigned long pen = 1; pen < set.size(); pen++)
               lines.push_back(image_window::overlay_line(d.part(set[pen]), d.part(set[pen-1]), color));

           // left eyebrow
           set = get_left_eyebrow(dets[i].num_parts());
           for (unsigned long pen = 1; pen < set.size(); pen++)
               lines.push_back(image_window::overlay_line(d.part(set[pen]), d.part(set[pen-1]), color));

           // Right eyebrow
           set = get_right_eyebrow(dets[i].num_parts());
           for (unsigned long pen = 1; pen < set.size(); pen++)
               lines.push_back(image_window::overlay_line(d.part(set[pen]), d.part(set[pen-1]), color));

           // Bottom part of the nose
           set = get_bottom_of_nose(dets[i].num_parts());
           for (unsigned long pen = 1; pen < set.size(); pen++)
               lines.push_back(image_window::overlay_line(d.part(set[pen]), d.part(set[pen-1]), color));

           // Line from the nose to the bottom part above already done above step

           // Left eye
           set = get_left_eye(dets[i].num_parts());
           for (unsigned long pen = 1; pen < set.size(); pen++)
               lines.push_back(image_window::overlay_line(d.part(set[pen]), d.part(set[pen-1]), color));

           // Right eye
           set = get_right_eye(dets[i].num_parts());
           for (unsigned long pen = 1; pen < set.size(); pen++)
               lines.push_back(image_window::overlay_line(d.part(set[pen]), d.part(set[pen-1]), color));

           // Lips outer part
           set = get_lips_outer_part(dets[i].num_parts());
           for (unsigned long pen = 1; pen < set.size(); pen++)
               lines.push_back(image_window::overlay_line(d.part(set[pen]), d.part(set[pen-1]), color));

           // Lips inside part
           set = get_lips_inside_part(dets[i].num_parts());
           for (unsigned long pen = 1; pen < set.size(); pen++)
               lines.push_back(image_window::overlay_line(d.part(set[pen]), d.part(set[pen-1]), color));
         }
return lines;
}


std::vector<image_window::overlay_circle> render_face_detections_noline(
    const std::vector<full_object_detection>& dets,
    const rgb_pixel color = rgb_pixel(0,255,0)
)
{
    std::vector<image_window::overlay_circle> circles;

    for (unsigned long i = 0; i < dets.size(); ++i)
    {
        const full_object_detection& d = dets[i];

        for (unsigned long pen = 0; pen < d.num_parts(); pen++)
        {
            unsigned long radius = 5;
            point center(d.part(pen));

            circles.push_back(
                image_window::overlay_circle(center, radius, color)
                );
        }
    }
    return circles;
}


std::vector<image_window::overlay_rect> render_face_detections_rect(
    const std::vector<full_object_detection>& dets,
    const rgb_pixel color = rgb_pixel(0,255,0)
)
{
    std::vector<image_window::overlay_rect> rects;
    for (unsigned long i = 0; i < dets.size(); i++)
    {
        rects.push_back(image_window::overlay_rect(
            dets[i].get_rect(),
            color)
            );
    }
    return rects;
}

std::vector<image_window::overlay_rect> render_face_detections_rect(
    const std::vector<rectangle>& dets,
    const rgb_pixel color = rgb_pixel(0,255,0)
)
{
    std::vector<image_window::overlay_rect> rects;
    for (unsigned long i = 0; i < dets.size(); i++)
    {
        rects.push_back(image_window::overlay_rect(dets[i], 
            color, 
            "face " + std::to_string(i)));
    }
    return rects;
}

//Todo :: get_face_chip_details
//        made that can work in 194 landmark
//Issue:: unknown what  Average positions of face points 17-67 doing


//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
void compute_pose_angle_form_pnp(
    cv::Mat img, 
    const full_object_detection& det)
{

    std::vector<double> rv(3), tv(3);
    cv::Mat rvec(rv),tvec(tv);
    cv::Vec3d eav;

    // Labelling the 3D Points derived from a 3D model of human face.
    // You may replace these points as per your custom 3D head model if any
    std::vector<cv::Point3f > modelPoints;
    modelPoints.push_back(cv::Point3f(2.37427,110.322,21.7776));    // l eye (v 314)
    modelPoints.push_back(cv::Point3f(70.0602,109.898,20.8234));    // r eye (v 0)
    modelPoints.push_back(cv::Point3f(36.8301,78.3185,52.0345));    //nose (v 1879)
    modelPoints.push_back(cv::Point3f(14.8498,51.0115,30.2378));    // l mouth (v 1502)
    modelPoints.push_back(cv::Point3f(58.1825,51.0115,29.6224));    // r mouth (v 695)
    modelPoints.push_back(cv::Point3f(-61.8886f,127.797,-89.4523f));  // l ear (v 2011)
    modelPoints.push_back(cv::Point3f(127.603,126.9,-83.9129f));     // r ear (v 1138)

    // labelling the position of corresponding feature points on the input image.
    // write the function return std::vector<cv::Point2f> srcImagePoints 
    // form std::vector<rectangle>& dets
    std::vector<cv::Point2f> srcImagePoints; 
    srcImagePoints.push_back(cv::Point2f(det.part(38).x(), det.part(38).y())); 
    srcImagePoints.push_back(cv::Point2f(det.part(44).x(), det.part(44).y()));
    srcImagePoints.push_back(cv::Point2f(det.part(30).x(), det.part(30).y()));    
    srcImagePoints.push_back(cv::Point2f(det.part(60).x(), det.part(60).y()));
    srcImagePoints.push_back(cv::Point2f(det.part(64).x(), det.part(64).y()));
    srcImagePoints.push_back(cv::Point2f(det.part(1).x(), det.part(1).y()));
    srcImagePoints.push_back(cv::Point2f(det.part(15).x(), det.part(15).y()));
 
    // processing find the relation form ip to op
    // ip: the 2D space point form alignment, op: the point form reference 3D model
    cv::Mat ip(srcImagePoints);
    cv::Mat op = cv::Mat(modelPoints);
    cv::Scalar m = cv::mean(cv::Mat(modelPoints));
    
    rvec = cv::Mat(rv);

    double _d[9] = {1,   0,   0,
                    0,  -1,   0,
                    0,   0,  -1};
    cv::Rodrigues( cv::Mat(3, 3, CV_64FC1, _d), rvec );
    tv[0]=0;
    tv[1]=0;
    tv[2]=1;
    tvec = cv::Mat( tv );

    double max_d = MAX(img.rows,img.cols);
    
    double _cm[9] = {max_d,     0,      (double)img.cols/2.0,
                         0, max_d,      (double)img.rows/2.0,
                         0,     0,                       1.0};
    cv::Mat camMatrix = cv::Mat(3, 3, CV_64FC1, _cm);

    double _dc[] = {0,0,0,0};
    cv::solvePnP(op, ip, camMatrix, cv::Mat(1,4,CV_64FC1,_dc), rvec, tvec, false, CV_EPNP);

    double rot[9] = {0};
    cv::Mat rotM(3, 3, CV_64FC1, rot);
    cv::Rodrigues( rvec, rotM );
    double* _r = rotM.ptr<double>();
    printf("rotation mat: \n %.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n",
       _r[0],_r[1],_r[2],_r[3],_r[4],_r[5],_r[6],_r[7],_r[8]);

    printf("trans vec: \n %.3f %.3f %.3f\n",tv[0],tv[1],tv[2]);

    double _pm[12] = {_r[0], _r[1], _r[2], tv[0],
                      _r[3], _r[4], _r[5], tv[1],
                      _r[6], _r[7], _r[8], tv[2]};
    cv::Mat tmp,tmp1,tmp2,tmp3,tmp4,tmp5;
    cv::decomposeProjectionMatrix(cv::Mat(3,4,CV_64FC1,_pm), tmp, tmp1, tmp2, tmp3, tmp4, tmp5, eav);

    printf("Face Rotation Angle:  %.5f %.5f %.5f\n", eav[0],eav[1],eav[2]);
}
//----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }
        
        if (argc == 1)
        {
            cout << "Call this program like this:" << endl;
            cout << "./webcam_face_pose_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
        }
        image_window win;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize(argv[1]) >> pose_model;

        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
            // Grab a frame
            cv::Mat temp;
            cap >> temp;
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<bgr_pixel> cimg(temp);

            // Detect faces 
            std::vector<rectangle> faces = detector(cimg);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes;

            //  time stamp before alignment
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            //  Get the alignment result for each face. 
            for (unsigned long i = 0; i < faces.size(); ++i)
                shapes.push_back(pose_model(cimg, faces[i]));
                


            //  time stamp after alignment
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Time difference = " << ( std::chrono::duration_cast<
                std::chrono::microseconds> (end - begin).count() ) << std::endl;


            for (unsigned long i = 0; i < shapes.size(); ++i)
                compute_pose_angle_form_pnp(temp, shapes[i]);

            // Get the initial cascade of alignment for each face.
            std::vector<full_object_detection> initial_shapes;
            for  (unsigned long i = 0; i < faces.size(); ++i)
                initial_shapes.push_back(
                    pose_model.get_initial_cascade(faces[i]));


            

            // Display it all on the screen
            win.clear_overlay();
            win.set_image(cimg);
            
            // Draw th rect from face detect result:faces
            if (faces.size() > 0 )
            {
                rgb_pixel color = rgb_pixel(255, 0, 0);
                win.add_overlay(render_face_detections_rect(faces, color));
            }

            if (initial_shapes.size() > 0)
            {
                rgb_pixel color = rgb_pixel(0, 0, 128);
                win.add_overlay(render_face_detections_noline(initial_shapes, color));
                win.add_overlay(render_face_detections_rect(initial_shapes, color));
            }

            // Draw the face alignment result from shapes
            if (shapes.size() > 0)
            {
                if (shapes[0].num_parts() == 68
                    || shapes[0].num_parts() == 194)
                    win.add_overlay(render_face_detections_adaptive(shapes));
                else
                    win.add_overlay(render_face_detections_noline(shapes));
                win.add_overlay(render_face_detections_rect(initial_shapes));
            }
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}

