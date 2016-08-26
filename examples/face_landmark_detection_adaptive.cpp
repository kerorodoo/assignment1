/*
    This application is modified form example:face_landmark_detection_ex
    The goal to made the application could run on landmark 194 case.

    The example:face_landmark_detection_ex 
    program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.
 
*/


#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>

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
            DLIB_CASSERT(dets[i].num_parts() == 68 || 
                         dets[i].num_parts() == 194,
                "\t std::vector<image_window::overlay_line> render_face_detections()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t dets["<< i <<"].num_parts():  " << dets[i].num_parts() 
            ); 
   
           const full_object_detection& d= dets[i];
           // Around Chin. Ear to Ear
           std::vector<int> set = get_around_chin_ear_to_ear_part(dets[i].num_parts());
           for (int pen = 1; pen < set.size(); pen++)
               lines.push_back(image_window::overlay_line(d.part(set[pen]), d.part(set[pen-1]), color));
           
           // Line on top of nose
           set = get_top_of_nose(dets[i].num_parts());
           for (int pen = 1; pen < set.size(); pen++)
               lines.push_back(image_window::overlay_line(d.part(set[pen]), d.part(set[pen-1]), color));
           
           // left eyebrow
           set = get_left_eyebrow(dets[i].num_parts());
           for (int pen = 1; pen < set.size(); pen++)
               lines.push_back(image_window::overlay_line(d.part(set[pen]), d.part(set[pen-1]), color));
           
           // Right eyebrow
           set = get_right_eyebrow(dets[i].num_parts());
           for (int pen = 1; pen < set.size(); pen++)
               lines.push_back(image_window::overlay_line(d.part(set[pen]), d.part(set[pen-1]), color));

           // Bottom part of the nose
           set = get_bottom_of_nose(dets[i].num_parts());
           for (int pen = 1; pen < set.size(); pen++)
               lines.push_back(image_window::overlay_line(d.part(set[pen]), d.part(set[pen-1]), color));
           
           // Line from the nose to the bottom part above already done above step
           
           // Left eye
           set = get_left_eye(dets[i].num_parts());
           for (int pen = 1; pen < set.size(); pen++)
               lines.push_back(image_window::overlay_line(d.part(set[pen]), d.part(set[pen-1]), color));
           
           // Right eye
           set = get_right_eye(dets[i].num_parts());
           for (int pen = 1; pen < set.size(); pen++)
               lines.push_back(image_window::overlay_line(d.part(set[pen]), d.part(set[pen-1]), color));
            
           // Lips outer part
           set = get_lips_outer_part(dets[i].num_parts());
           for (int pen = 1; pen < set.size(); pen++)
               lines.push_back(image_window::overlay_line(d.part(set[pen]), d.part(set[pen-1]), color));

           // Lips inside part
           set = get_lips_inside_part(dets[i].num_parts());
           for (int pen = 1; pen < set.size(); pen++)
               lines.push_back(image_window::overlay_line(d.part(set[pen]), d.part(set[pen-1]), color));
         }
return lines;
}

//Todo :: get_face_chip_details
//        made that can work in 194 landmark
//Issue:: unknown what  Average positions of face points 17-67 doing
chip_details get_chip_details (
        const full_object_detection& det,
        const unsigned long size = 200
    )
{
    return chip_details(det.get_rect(), chip_dims(size,size));   
}
std::vector<chip_details> get_face_chip_details_adaptive (
        const std::vector<full_object_detection>& dets,
        const unsigned long size = 200,
        const double padding = 0.2
)
{
    std::vector<chip_details> res;
    res.reserve(dets.size());
    for (unsigned long i = 0; i < dets.size(); ++i)
        res.push_back(get_chip_details(dets[i], size));
    return res;
}
//----------------------------------------------------------------------------------------




// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  
    try
    {
        // This example takes in a shape model file and then a list of images to
        // process.  We will take these filenames in as command line arguments.
        // Dlib comes with example images in the examples/faces folder so give
        // those as arguments to this program.
        if (argc == 1)
        {
            cout << "Call this program like this:" << endl;
            cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
            cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
            cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
            cout << "\nYou can get the shape_predictor_194_face_landmarks.dat file from:\n";
            cout << "http://stackoverflow.com/questions/36711905/dlib-train-shape-predictor-ex-cpp" << endl;
            return 0;
        }

        // We need a face detector.  We will use this to get bounding boxes for
        // each face in an image.
        frontal_face_detector detector = get_frontal_face_detector();
        // And we also need a shape_predictor.  This is the tool that will predict face
        // landmark positions given an image and face bounding box.  Here we are just
        // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
        // as a command line argument.
        
        shape_predictor sp;
        deserialize(argv[1]) >> sp;


        image_window win, win_faces;
        // Loop over all the images provided on the command line.
        for (int i = 2; i < argc; ++i)
        {
            cout << "processing image " << argv[i] << endl;
            array2d<rgb_pixel> img;
            load_image(img, argv[i]);
            // Make the image larger so we can detect small faces.
            pyramid_up(img);

            // Now tell the face detector to give us a list of bounding boxes
            // around all the faces in the image.
            std::vector<rectangle> dets = detector(img);
            cout << "Number of faces detected: " << dets.size() << endl;

            // Now we will go ask the shape_predictor to tell us the pose of
            // each face we detected.
            
            std::vector<full_object_detection> shapes;
            for (unsigned long j = 0; j < dets.size(); ++j)
            {
                full_object_detection shape = sp(img, dets[j]);
                cout << "number of parts: "<< shape.num_parts() << endl;
                cout << "pixel position of first part:  " << shape.part(0) << endl;
                cout << "pixel position of second part: " << shape.part(1) << endl;
                // You get the idea, you can get all the face part locations if
                // you want them.  Here we just store them in shapes so we can
                // put them on the screen.
            }
            

            // Now let's view our face poses on the screen.
            win.clear_overlay();
            win.set_image(img);
            win.add_overlay(render_face_detections_adaptive(shapes));

            // We can also extract copies of each face that are cropped, rotated upright,
            // and scaled to a standard size as shown here:
            dlib::array<array2d<rgb_pixel> > face_chips;
            extract_image_chips(img, get_face_chip_details_adaptive(shapes), face_chips);
            win_faces.set_image(tile_images(face_chips));

            cout << "Hit enter to process the next image..." << endl;
            cin.get();
        }
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

