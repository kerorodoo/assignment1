#include <dlib/opencv.h>

#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core_c.h>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <dlib/dir_nav.h>
#include <dlib/misc_api.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <time.h>
#include <dlib/string.h>

using namespace dlib;
using namespace std;
using namespace cv;

/*-------------------------------------------------------------------*/
 
chip_details get_face_chip_details (
    const full_object_detection& final_det,
    const full_object_detection& init_det,
    const unsigned long size = 200,
    const double padding = 0.2
    )
{
    DLIB_CASSERT(final_det.num_parts() == init_det.num_parts(),
            "\t chip_details get_face_chip_details()"
            << "\n\t Invalid inputs were given to this function."
            << "\n\t final_det.num_parts: " << final_det.num_parts()
            << "\n\t init_det.num_parts: " << init_det.num_parts()
            );


    std::vector<dlib::vector<double,2> > from_points, to_points;
    for (unsigned long i = 0; i < final_det.num_parts(); i++)
    {
            dlib::vector<double,2> p;
            double x = ( 
                static_cast<double>( init_det.part(i).x() )
                - static_cast<double>( init_det.get_rect().left() ) 
                ) / static_cast<double>( init_det.get_rect().width() );
            double y = ( 
                static_cast<double>( init_det.part(i).y() )
                - static_cast<double>( init_det.get_rect().top() ) 
                ) / static_cast<double>( init_det.get_rect().height() );

            p.x() = (padding + x) / (2*padding+1);
            p.y() = (padding + y) / (2*padding+1);
            from_points.push_back(p * size);
            to_points.push_back(final_det.part(i));
    }
    return chip_details(from_points, to_points, chip_dims(size,size));
}

std::vector<chip_details> get_face_chip_details (
    const std::vector<full_object_detection>& final_dets,
    const std::vector<full_object_detection>& init_dets,
    const unsigned long size = 200,
    const double padding = 0.2
    )
{
    std::vector<chip_details> res;
    res.reserve(final_dets.size());
    for (unsigned long i = 0; i < final_dets.size(); ++i)
        res.push_back(get_face_chip_details(final_dets[i], init_dets[i], size, padding));
    return res;
}

/*--------------------------------------------------------------*/

int main(int argc, char const *argv[])
{
	if (argc == 1)
	{
		cout << "Call this program like this:" << endl;
		cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat face_dataset_folder" << endl;
		cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
		cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		return 0;
	}

	//Count time
	double START, END;
	START = clock();

	//Create alignment face directory
	string alignment_face = get_current_dir() + "/Alignment-SWD-Faces";
	create_directory(alignment_face);
	directory alignment_face_dirs(alignment_face);
	cout << "Alignment Face will store at " << alignment_face_dirs.full_name() << endl;

	frontal_face_detector detector = get_frontal_face_detector();

	shape_predictor sp;
	deserialize(argv[1]) >> sp;

	image_window win, win_faces;
	string dataset_loc;
	std::vector<directory> image_dirs;
	std::vector<file> images;
	array2d<rgb_pixel> img;

	//count skip image
	int skip_image_counter = 0;
	std::vector<string> skip_images_name;

	// Loop over all the images provided on the command line.

	for (int i = 2; i < argc; ++i)
	{

		cout << "processing image folder " << argv[i] << endl;
		dataset_loc = argv[i];

		//Get Images directories
		directory dataset_dirs(dataset_loc);

		cout << "dataset directory: " << dataset_dirs.name() << endl;
		cout << "dataset full path: " << dataset_dirs.full_name() << endl;
		cout << "dataset is root:   " << ((dataset_dirs.is_root())?"yes":"no") << endl;

		image_dirs = dataset_dirs.get_dirs();
		cout << "Number of Image Directories in this Dataset: " << image_dirs.size() << endl;

		//Each Identity folder
		for(unsigned long x = 0; x < image_dirs.size(); x++){

			//Browser directory
			cout << "Browsing Image Directory: " << image_dirs[x].full_name() << endl;
			string dir_name = image_dirs[x].name();
			create_directory(alignment_face + "/" + dir_name);
			images = image_dirs[x].get_files();
			cout << "Number of Images in current DIrectory: " << images.size() << endl;

			//Each Image
			for(unsigned long y = 0; y < images.size(); y++){
				//Load image
				string image_name = images[y].name();
				cout << "Image Name: " << images[y].name() << "\n";
				cout << "Loading Image: " << images[y].full_name() << "\n";
				cout << "JPEG_Loader: " << images[y] << "\n";
				load_image(img, images[y]);
				// Make the image larger so we can detect small faces.
				pyramid_up(img);

				// Now tell the face detector to give us a list of bounding boxes
				// around all the faces in the image.
				std::vector<dlib::rectangle> dets = detector(img);
				cout << "Number of faces detected: " << dets.size() << endl;

				//Skip this image if no Face detected
				int max_face = 0;
				int max_face_idx = 0;
				if(dets.size() == 0){
					cout << "Skip Image: " << image_name << endl;
					skip_images_name.push_back(image_name);
					skip_image_counter++;
					continue;
				}else if(dets.size() > 1){
					//cout << "Hit enter to process the next action..." << endl;
					//cin.get();
					for(unsigned int z = 0; z < dets.size(); z++ ){
						//Caculate Face size
						cout << "Face Rectangle Size -> width: " << dets[z].width() << " height: " << dets[z].height() << endl;
						int face_area = dets[z].width() * dets[z].height();
						if( face_area > max_face ){
							max_face = face_area;
							max_face_idx = z;
						}
					}
				}

				// Now we will go ask the shape_predictor to tell us the pose of
				// each face we detected.
				std::vector<full_object_detection> shapes;
				//extract face config
				unsigned long size = 60;
				double padding = 0.005;


				full_object_detection shape = sp(img, dets[max_face_idx]);
				cout << "number of parts: "<< shape.num_parts() << endl;
				cout << "pixel position of first part:  " << shape.part(0) << endl;
				cout << "pixel position of second part: " << shape.part(1) << endl;

				shapes.push_back(shape);

                // Get the initial cascade of alignment for each face.
                std::vector<full_object_detection> initial_shapes;
                for  (unsigned long i = 0; i < dets.size(); ++i)
                    initial_shapes.push_back(sp.get_initial_cascade(dets[i]));

				// Now let's view our face poses on the screen.
				win.clear_overlay();
				win.set_image(img);
				win.add_overlay(dets);
				win.add_overlay(render_face_detections(shapes));

				// We can also extract copies of each face that are cropped, rotated upright,
				// and scaled to a standard size as shown here:
				dlib::array<array2d<rgb_pixel> > face_chips;
				extract_image_chips(img, get_face_chip_details(shapes, initial_shapes, size, padding), face_chips);
				win_faces.set_image(tile_images(face_chips));

				//Save JPG image
				save_jpeg(tile_images(face_chips), (alignment_face + "/" + dir_name + "/" + image_name), 90);

			}

		}

	}

	//Show how many images is skipped
	cout << "Skip " << skip_image_counter << " Images !!" << endl;

	//Show skipped images name
	for(unsigned long z = 0; z < skip_images_name.size(); z++){
		cout << "Skipped Image: " << skip_images_name[z] << endl;
	}

	//Count time
	END = clock();
	cout << endl << "Computing take: " << (END - START) / CLOCKS_PER_SEC << " s" << endl;
}

