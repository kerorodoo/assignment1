// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the support vector machine
    utilities from the dlib C++ Library.  

    This example creates a simple set of local binary pattern form faces folder.
    
*/

#include <vector>
#include <iostream>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_transforms/lbp.h>
#include <dlib/image_processing.h>
#include <dlib/statistics.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>
#include <dlib/algs.h>

#include <dlib/svm.h>

using namespace std;
using namespace dlib;

template <
    typename sample_type
    >
void extract_samples_form_folder (
    frontal_face_detector detector,
    shape_predictor sp,
    directory dirs,
    std::vector<sample_type>& samples,
    std::vector<double>& labels,
    double label = +1
)
{
    std::vector<directory> image_dirs;  //path of folder form input dirs
    std::vector<file> images;   //path of images
    dlib::array2d<rgb_pixel> img;
    
    cout << "dataset directory: " << dirs.name() << endl;
	cout << "dataset full path: " << dirs.full_name() << endl;
	cout << "dataset is root:   " << ((dirs.is_root())?"yes":"no") << endl;

	image_dirs = dirs.get_dirs();
	cout << "Number of Image Directories in this Dataset: " << image_dirs.size() << endl;

    //Each Identity folder
	for(unsigned long x = 0; x < image_dirs.size(); x++)
    {
		//Browser directory
		cout << "Browsing Image Directory: " << image_dirs[x].full_name() << endl;
		string dir_name = image_dirs[x].name();
		cout << "Identity folder: " << dir_name << endl;
		images = image_dirs[x].get_files();
		cout << "Number of Images in current DIrectory: " << images.size() << endl;
    
        //Each Image
		for(unsigned long y = 0; y < images.size(); y++)
        {
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


            for (unsigned long i = 0; i < dets.size(); i++)
            {
                // Now we will go ask the shape_predictor to tell us the pose of
				// each face we detected.
				std::vector<full_object_detection> shapes;

				full_object_detection shape = sp(img, dets[i]);
				cout << "number of parts: "<< shape.num_parts() << endl;
				cout << "pixel position of first part:  " << shape.part(0) << endl;
				cout << "pixel position of second part: " << shape.part(1) << endl;

                std::vector<double> feats;

                extract_highdim_face_lbp_descriptors(img, shape, feats);

                cout << "feats: " << feats.size() << endl;

                sample_type sample(feats.size(), 1);

                for (unsigned long feats_idx = 0; feats_idx < feats.size(); feats_idx++)
                {
                    sample(feats_idx) = feats[feats_idx];
                }

                DLIB_CASSERT(sample.size() == 99120,
                "\t void extract_samples_form_folder()"
                << "\n\t Invalid result to this function."
                << "\n\t sample.size(): " << sample.size()
                );
                
                samples.push_back(sample);
                labels.push_back(label);
            }
        }
    }
}


int main(int argc, char const *argv[])
{
    // This typedef declares a matrix with 2 rows and 1 column.  It will be the
    // object that contains each of our 2 dimensional samples.   (Note that if
    // you wanted more than 2 features in this vector you can simply change the
    // 2 to something else.  Or if you don't know how many features you want
    // until runtime then you can put a 0 here and use the matrix.set_size()
    // member function)
    typedef matrix<double, 0, 1> sample_type;

    // This is a typedef for the type of kernel we are going to use in this
    // example.  In this case I have selected the radial basis kernel that can
    // operate on our 2D sample_type objects.  You can use your own custom
    // kernels with these tools as well, see custom_trainer_ex.cpp for an
    // example.
    typedef radial_basis_kernel<sample_type> kernel_type;

	if (argc == 1)
	{
		cout << "Call this program like this:" << endl;
		cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat face_dataset_folder" << endl;
		cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
		cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		return 0;
	}

    //  initial the face detector
    frontal_face_detector detector = get_frontal_face_detector();

    //  initial the shape predictor 
    shape_predictor sp;
	deserialize(argv[1]) >> sp;


    //  initial the faces folder.
    //  The faces directory contains a training dataset.
    const std::string faces_directory = argv[2];

    
    //std::vector<std::vector<full_object_detection> > faces_positive, faces_negative;

    // Now we make objects to contain our samples and their respective labels.
    std::vector<sample_type> samples;
    std::vector<double> labels;

    std::vector<file> images;
    
    //  Get positive sample form positive_samples directories
    directory positive_dirs(faces_directory + "/positive_samples");
    extract_samples_form_folder(detector, sp, positive_dirs, samples, labels, +1);
    cout << "\tcomplete extract positive sample: " << endl;
    cout << "\ttotal sample size: " << samples.size() << endl;


    //  Get negative sample form negative_samples directories
    directory negative_dirs(faces_directory + "/negative_samples");
    extract_samples_form_folder(detector, sp, negative_dirs, samples, labels, -1);
    cout << "\tcomplete extract negative sample: " << endl;
    cout << "\ttotal sample size: " << samples.size() << endl;

    // Here we normalize all the samples by subtracting their mean and dividing
    // by their standard deviation.  This is generally a good idea since it
    // often heads off numerical stability problems and also prevents one large
    // feature from smothering others.  Doing this doesn't matter much in this
    // example so I'm just doing this here so you can see an easy way to
    // accomplish it.  
    vector_normalizer<sample_type> normalizer;
    // Let the normalizer learn the mean and standard deviation of the samples.
    normalizer.train(samples);
    // now normalize each sample
    cout << "normalizing all the samples ..." << endl;
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]);
    cout << "\tsamples size: " << samples.size() << endl;
    cout << "\tlabels size: " << labels.size() << endl;

    // Now that we have some data we want to train on it.  However, there are
    // two parameters to the training.  These are the C and gamma parameters.
    // Our choice for these parameters will influence how good the resulting
    // decision function is.  To test how good a particular choice of these
    // parameters are we can use the cross_validate_trainer() function to perform
    // n-fold cross validation on our training data.  However, there is a
    // problem with the way we have sampled our distribution above.  The problem
    // is that there is a definite ordering to the samples.  That is, the first
    // half of the samples look like they are from a different distribution than
    // the second half.  This would screw up the cross validation process but we
    // can fix it by randomizing the order of the samples with the following
    // function call.
    cout << "randomizing all the samples ..." << endl;
    randomize_samples(samples, labels);
    cout << "\tsamples size: " << samples.size() << endl;
    cout << "\tlabels size: " << labels.size() << endl;


    // here we make an instance of the svm_c_trainer object that uses our kernel
    // type.
    svm_c_trainer<kernel_type> trainer;

    // Now we loop over some different C and gamma values to see how good they
    // are.  Note that this is a very simple way to try out a few possible
    // parameter choices.  You should look at the model_selection_ex.cpp program
    // for examples of more sophisticated strategies for determining good
    // parameter choices.
    /*cout << "doing cross validation ..." << endl;
    for (double gamma = 0.00001; gamma <= 1; gamma *= 5)
    {
        for (double C = 1; C < 100000; C *= 5)
        {
            // tell the trainer the parameters we want to use
            trainer.set_kernel(kernel_type(gamma));
            trainer.set_c(C);

            cout << "gamma: " << gamma << "    C: " << C;
            // Print out the cross validation accuracy for 3-fold cross validation using
            // the current gamma and C.  cross_validate_trainer() returns a row vector.
            // The first element of the vector is the fraction of +1 training examples
            // correctly classified and the second number is the fraction of -1 training
            // examples correctly classified.
            cout << "     cross validation accuracy: " 
                 << cross_validate_trainer(trainer, samples, labels, 3);
        }
    }
    */

    // From looking at the output of the above loop it turns out that good
    // values for C and gamma for this problem are 5 and 0.15625 respectively.
    // So that is what we will use.

    // Now we train on the full set of data and obtain the resulting decision
    // function.  The decision function will return values >= 0 for samples it
    // predicts are in the +1 class and numbers < 0 for samples it predicts to
    // be in the -1 class.
    // doing cross validation ...
    // gamma: 1e-05    C: 1     cross validation accuracy:        1 0.996363 
    // gamma: 1e-05    C: 5     cross validation accuracy:        1 0.996363 
    // gamma: 1e-05    C: 25     cross validation accuracy:        1 0.996363 
    // gamma: 1e-05    C: 125     cross validation accuracy:        1 0.996363 
    trainer.set_kernel(kernel_type(0.00001));
    trainer.set_c(5);
    
    
    typedef decision_function<kernel_type> dec_funct_type;
    typedef normalized_function<dec_funct_type> funct_type;

    // Here we are making an instance of the normalized_function object.  This
    // object provides a convenient way to store the vector normalization
    // information along with the decision function we are going to learn.  
    funct_type learned_function;
    learned_function.normalizer = normalizer;  // save normalization information
    learned_function.function = trainer.train(samples, labels); // perform the actual SVM training and save the results

    // print out the number of support vectors in the resulting decision function
    cout << "\nnumber of support vectors in our learned_function is " 
         << learned_function.function.basis_vectors.size() << endl;
    
    serialize("saved_function.dat") << learned_function;
    
    
    /*
    // We can also train a decision function that reports a well conditioned
    // probability instead of just a number > 0 for the +1 class and < 0 for the
    // -1 class.  An example of doing that follows:
    typedef probabilistic_decision_function<kernel_type> probabilistic_funct_type;  
    typedef normalized_function<probabilistic_funct_type> pfunct_type;

    pfunct_type learned_pfunct; 
    learned_pfunct.normalizer = normalizer;
    learned_pfunct.function = train_probabilistic_decision_function(trainer, samples, labels, 3);
    // Now we have a function that returns the probability that a given sample is of the +1 class.

    // print out the number of support vectors in the resulting decision function
    //cout << "\nnumber of support vectors in our learned_pfunct is " 
    //     << learned_pfunct.function.basis_vectors.size() << endl;

    // Another thing that is worth knowing is that just about everything in dlib
    // is serializable.  So for example, you can save the learned_pfunct object
    // to disk and recall it later like so:
    serialize("saved_pfunction.dat") << learned_pfunct;
    */
}
