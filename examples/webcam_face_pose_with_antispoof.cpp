// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    

    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead 
    of files.


*/

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms/lbp.h>
#include <dlib/svm.h>

#include "swd_tools.cpp"

#include <string>
#include <chrono>


using namespace dlib;
using namespace std;


//----------------------------------------------------------------------------------------


template <
    typename image_type,
    typename T
    >
void extract_customize_descriptors(
    const image_type& img_rgb,
    const full_object_detection& det,
    std::vector<T>& feats
)
{
    feats.clear();

    dlib::array2d<double> img_gray;
    dlib::assign_image(img_gray, img_rgb);

    dlib::array2d<double> img_diff;
    anisodiff(img_gray, img_diff, 5, 100, 0.25, 2);

    extract_customize_lbp_descriptors(img_diff, det, feats, 3);
}

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


//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    typedef matrix<double, 0, 1> sample_type;
    typedef radial_basis_kernel<sample_type> kernel_type;

    try
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }
        
        if (argc != 4)
        {
            cout << "Call this program like this:" << endl;
            cout << "./webcam_face_pose_ex shape_predictor_68_face_landmarks.dat pu.dat saved_pfunction.dat faces/*.jpg" << endl;
        }
        image_window win, win_hist;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize(argv[1]) >> pose_model;

        // Load svd_model for reduce high dimension feature
        matrix<double> pu;
        deserialize(argv[2]) >> pu;

        // Load svm model for anti-spoo
        //if (strcmp(argv[2], "_d") > 0)
        typedef decision_function<kernel_type> nfunct_type;
        //else
        //typedef probabilistic_decision_function<kernel_type> nfunct_type;  
        typedef normalized_function<nfunct_type> funct_type;
        
        funct_type learned_funct;
        deserialize(argv[3]) >> learned_funct;

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
            //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            //  Get the alignment result for each face. 
            for (unsigned long i = 0; i < faces.size(); ++i)
                shapes.push_back(pose_model(cimg, faces[i]));

            //  time stamp after alignment
            //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            //std::cout << "Time difference (for shape predictor)= " << ( std::chrono::duration_cast<
            //    std::chrono::microseconds> (end - begin).count() ) << std::endl;

            // Get the initial cascade of alignment for each face.
            std::vector<full_object_detection> initial_shapes;
            for  (unsigned long i = 0; i < faces.size(); ++i)
                initial_shapes.push_back(
                    pose_model.get_initial_cascade(faces[i]));

            
            dlib::array2d<rgb_pixel> img;
            assign_image(img, cimg);


            std::vector<bool> positive_flag;
            //  time stamp before classifier
            //begin = std::chrono::steady_clock::now();
           
            // extract highdim face lbp descriptors and Get classifier output
            std::vector<double> feats;

            dlib::array2d<double> hist_img;
            std::vector<sample_type> samples;
            
            
            for (unsigned long i = 0; i < faces.size(); ++i)
            {
                dlib::array2d<rgb_pixel> face_chips;
                extract_image_chip(img, get_face_chip_details(shapes[i]), face_chips);

                extract_customize_descriptors(face_chips, shapes[i], feats);
                
                sample_type sample(feats.size(), 1);
                for (unsigned long feats_idx = 0; feats_idx < feats.size(); feats_idx++)
                {
                    sample(feats_idx) = feats[feats_idx];
                }

                samples.push_back(sample);

        
                //matrix<double> samp(trans(sample) * pu);
                matrix<double> samp(sample);
                double classifier_output = learned_funct(samp);

                cout << "highdim face lbp descriptors feats after reduce: " << samp.nr()  << "x" << samp.nc() << endl;

                cout << "face[" << i << "] the classifier output is " << classifier_output << endl;
                if (classifier_output > 0.5)
                {
                    cout << "positive !!!" << endl;
                    positive_flag.push_back(true);

                }
                else
                    positive_flag.push_back(false);
            }


            hist_img.set_size(faces.size()*20, feats.size());
           
            if (samples.size() == 2)
            {
                for (unsigned long sample_idx = 0; sample_idx < samples[0].nr(); sample_idx++)
                    if (samples[0](0, sample_idx) == samples[1](0, sample_idx))
                        cout << "samples[0](0, sample_idx) != samples[1](0, sample_idx)" << endl; 
            }

            for (unsigned long i = 0; i < faces.size(); ++i)
            {
                if(i > samples.size())
                    continue;
                for (unsigned long sample_idx = 0; sample_idx < samples[i].nr(); sample_idx++)
                {
                    for (unsigned long r = 0; r < 20; r++)
                        dlib::assign_pixel(hist_img[i * 20 + r][sample_idx], samples[i](0, sample_idx));
                
                }
            }

            //end = std::chrono::steady_clock::now();
            //std::cout << "Time difference (for classifier)= " << ( std::chrono::duration_cast<
            //    std::chrono::microseconds> (end - begin).count() ) << std::endl;
            // 8ms in test

            // Display it all on the screen
            win.clear_overlay();
            win.set_image(cimg);

            win_hist.clear_overlay();
            win_hist.set_image(hist_img);
            
            // Draw th rect from face detect result:faces
            if (faces.size() > 0 )
            {
                for (unsigned long i = 0; i < faces.size(); ++i)
                {
                    std::vector<rectangle> face;
                    rgb_pixel color = rgb_pixel(128, 128, 128);
                    if (positive_flag[i])
                        color = rgb_pixel(255, 0, 0);
                    face.push_back(faces[i]);
                    win.add_overlay(render_face_detections_rect(face, color));
                }
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

