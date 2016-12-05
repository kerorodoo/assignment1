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
#include <dlib/statistics/dpca.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>
#include <dlib/matrix.h>

#include <dlib/svm.h>

using namespace std;
using namespace dlib;


//-----------------------------------------------------------------------------

template <
    typename image_type,
    typename T
    >
void extract_customize_lbp_descriptors (
    const image_type& img,
    const full_object_detection& det, 
    std::vector<T>& feats,
    const unsigned long num_scales = 5
)
{
    //const unsigned long num_scales = 5; 
    feats.clear();

    array2d<rgb_pixel> img_fixed(60, 60);
    resize_image(img, img_fixed);

    array2d<unsigned char> lbp;
    make_uniform_lbp_image(img_fixed, lbp);

    std::vector<point> parts;
    parts.push_back(det.part(30));

    for (unsigned long i = 0; i < parts.size(); ++i)
        //extract_histogram_descriptors(lbp, parts[i], feats, 9, 9);
        extract_uniform_lbp_descriptors (lbp, feats, 20);

    if (num_scales > 1)
    {
        pyramid_down<4> pyr;

        image_type img_temp;
        pyr(img_fixed, img_temp);

        unsigned long num_pyr_calls = 1;

        // now pull the features out at coarser scales
        for (unsigned long iter = 1; iter < num_scales; ++iter)
        {
	    // now do the feature extraction
	    make_uniform_lbp_image(img_temp, lbp);
	    std::cout << "\n\t\t img_pyr:" << img_temp.nr() 
                      << "x" << img_temp.nc();

            for (unsigned long i = 0; i < parts.size(); ++i)
                    //extract_histogram_descriptors(lbp, pyr.point_down(parts[i],num_pyr_calls), feats, 10, 10);
		    extract_uniform_lbp_descriptors (lbp, feats, 20);


            if (iter+1 < num_scales)
            {
                pyr(img_temp);
                //pyramid_up(img_temp, pyr);
                ++num_pyr_calls;
            }
        }
    }

    for (unsigned long i = 0; i < feats.size(); ++i)
	feats[i] = std::sqrt(feats[i]);
    //DLIB_ASSERT(feats.size() == 99120, feats.size());
}


//-----------------------------------------------------------------------------
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

		
                extract_customize_lbp_descriptors(img, shape, feats, 1);

                cout << "feats: " << feats.size() << endl;

                sample_type sample(feats.size(), 1);

                for (unsigned long feats_idx = 0; feats_idx < feats.size(); feats_idx++)
                {
                    sample(feats_idx) = feats[feats_idx];
                }

                //DLIB_CASSERT(sample.size() == 99120,
                //"\t void extract_samples_form_folder()"
                //<< "\n\t Invalid result to this function."
                //<< "\n\t sample.size(): " << sample.size()
                //);
                
                samples.push_back(sample);
                labels.push_back(label);
            }
        }
    }
}

template <
    typename sample_type,
    typename lower_sample_type
    >
void reduce_samples_dimension(
    std::vector<sample_type>& samples,
    std::vector<lower_sample_type>& dpca_samples,
    const double eps
    )
{
    discriminant_pca<sample_type> dpca;

    cout << "\n\tvoid reduce_samples_dimension start: " << samples.size() << endl;

    matrix<double> m(samples[0].nr(), samples.size());

    for (int i = 0; i < m.nc(); ++i)
    {

        set_colm(m, i) = samples[i];

        samples[i].set_size(0,0);   // free RAM

        //DLIB_CASSERT(dpca.in_vector_size() == 99120,
        //    "\t void discriminant_pca()"
        //    << "\n\t Invalid result to this function."
        //    << "\n\t dpca.in_vector_size(): " << dpca.in_vector_size()
        //);
    }

    samples.resize(0); //free RAM

    cout << "\n\tvoid reduce_samples_dimension before reduce: ";
    cout << "\n\t   m.size: " << m.nr() << "x" << m.nc();

    // Do SVD to reduce dims
    matrix<double> pu,pw,pv;
    
    string pu_path = "pu.dat";
    if (!file_exists(pu_path))
    {
        cout << "\n\tpu.dat  not exists compute svd_fast ... ";
        cout << "\n\tstaring svd_fast ... ";
    
        svd_fast(m, pu, pw, pv, 10000, 4);
    
        serialize("pw.dat") << pw;
        serialize("pv.dat") << pv;
        serialize("pu.dat") << pu;
        //dpca_samples = m*pv;


        cout << "\n\tcomplete svd_fast ... ";
        cout << "\n\t   m.size: " << m.nr() << "x" << m.nc(); //mxn
        cout << "\n\t   pu.size: " << pu.nr() << "x" << pu.nc(); //mxk
        cout << "\n\t   pw.size: " << pw.nr() << "x" << pw.nc();  //kx1
        cout << "\n\t   pv.size: " << pv.nr() << "x" << pv.nc();  //nxk

        pw.set_size(0,0);   // free RAM
        pv.set_size(0,0);   // free RAM
    }

    deserialize(pu_path) >> pu;
    cout << "\n\t   pu.size: " << pu.nr() << "x" << pu.nc(); //mxn
    
    cout << "\n\tvoid reduce_samples_dimension after reduce: ";
    cout << "\n\t   reduce samples ..." << endl;
    for (int i = 0; i < m.nc(); ++i)
    {
        //matrix<double> samp(trans(trans(colm(m,i)) * pu));
        matrix<double> samp(colm(m,i));
        cout << "\n\t   samp.size: " << samp.nr() << "x" << samp.nc();
        cout << "\n\t   samp:" << trans(samp);
        dpca_samples.push_back(samp);

    }

    cout << "\n\t   dpca_samples.size: " << dpca_samples[0].nr() << "x" << dpca_samples.size() << endl;  //nxn
    
}


template <
    typename funct_type
    >
void get_accurary_cross_training_set (
    frontal_face_detector detector,
    shape_predictor sp,
    directory dirs,
    normalized_function<funct_type> learned_funct,
    double label = +1
)
{
    typedef matrix<double, 0, 1> sample_type;
    
    std::vector<directory> image_dirs;  //path of folder form input dirs
    std::vector<file> images;   //path of images
    dlib::array2d<rgb_pixel> img;
    

	image_dirs = dirs.get_dirs();

    // getting pu
    string pu_path = "pu.dat";
    matrix<double> pu;
    deserialize(pu_path) >> pu;
    
    int image_count = 0;
    int true_count= 0;

    //Each Identity folder
    for(unsigned long x = 0; x < image_dirs.size(); x++)
    {
       string dir_name = image_dirs[x].name(); //Identity folder
		
       images = image_dirs[x].get_files();     //Images path
    
        //Each Image
        for(unsigned long y = 0; y < images.size(); y++)
        {
	    //Load image
	    string image_name = images[y].name();
	    load_image(img, images[y]);

            // Make the image larger so we can detect small faces.
	    pyramid_up(img);

	    // Now tell the face detector to give us a list of bounding boxes
	    // around all the faces in the image.
            // detect faces
	    std::vector<dlib::rectangle> dets = detector(img);


            for (unsigned long i = 0; i < dets.size(); i++)
            {

                image_count++;

                // Now we will go ask the shape_predictor to tell us the pose of
		// each face we detected.
		std::vector<full_object_detection> shapes;

                // detect landmarks
	        full_object_detection shape = sp(img, dets[i]);
				
                std::vector<double> feats;

                //extract faces
                extract_customize_lbp_descriptors(img, shape, feats, 1);


                sample_type sample(feats.size(), 1);

                for (double feats_idx = 0; feats_idx < feats.size(); feats_idx++)
                {
                    sample(feats_idx) = feats[feats_idx];
                }

                
                //matrix<double> samp(trans(trans(sample) * pu));
                matrix<double> samp(sample);
                
                double classifier_output = learned_funct(samp);

                //cout << "\n\tthe classifier_output from: " 
                //     << dirs << "/" << dir_name << "/" << image_name;
                //cout << "\n\tthe classifier_output is: " << classifier_output;
                if (label > 0 && classifier_output > 0.5)
                    true_count++;
                else if (classifier_output < 0 || classifier_output < 0.05)
                    true_count++;
            }
        }
    }

    cout << "\n\tThe accuracy of " << dirs.name() << " is : " 
         << (double)true_count / (double)image_count << endl;

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
    typedef matrix<double, 0, 1> lower_sample_type;

    // This is a typedef for the type of kernel we are going to use in this
    // example.  In this case I have selected the radial basis kernel that can
    // operate on our 2D sample_type objects.  You can use your own custom
    // kernels with these tools as well, see custom_trainer_ex.cpp for an
    // example.
    typedef radial_basis_kernel<lower_sample_type> kernel_type;

	if (argc == 1)
	{
		cout << "Call this program like this:" << endl;
		cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat spoof_face_dataset_folder" << endl;
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
    std::vector<lower_sample_type> dpca_samples;
    std::vector<double> labels;

    std::vector<file> images;
    
    //  Get positive sample form train_positive_samples directories
    directory train_positive_dirs(faces_directory + "/train/positive_samples");
    extract_samples_form_folder(detector, sp, train_positive_dirs, samples, labels, +1);
    cout << "\n\tcomplete extract train positive sample: ";
    cout << "\n\ttotal sample in folder: " << samples.size();


    //  Get negative sample form train_negative_samples directories
    directory train_negative_dirs(faces_directory + "/train/negative_samples");
    extract_samples_form_folder(detector, sp, train_negative_dirs, samples, labels, -1);
    cout << "\n\tcomplete extract negative sample: ";
    cout << "\n\ttotal sample in folder: " << samples.size() << endl;

    // Get test sample form test_positive_samples directories
    directory test_positive_dirs(faces_directory + "/test/positive_samples");
    // Get test sample form test_negative_samples directories
    directory test_negative_dirs(faces_directory + "/test/negative_samples");


    // Reduce dimension of samples by dpca
    cout << "\nreduceing dimension of samples ...";
    reduce_samples_dimension(samples, dpca_samples, 0.9);
    cout << "\n\tcomplete reduce samples dimension.";
    cout << "\n\ttotal dpca_sample size: " << dpca_samples.size(); 
    cout << "\n\teach dpca_sample size: " << dpca_samples[0].nr() << "x" 
					<< dpca_samples[0].nc() << endl;

    // Here we normalize all the samples by subtracting their mean and dividing
    // by their standard deviation.  This is generally a good idea since it
    // often heads off numerical stability problems and also prevents one large
    // feature from smothering others.  Doing this doesn't matter much in this
    // example so I'm just doing this here so you can see an easy way to
    // accomplish it.  
    vector_normalizer<lower_sample_type> normalizer;
    // Let the normalizer learn the mean and standard deviation of the samples.
    normalizer.train(dpca_samples);
    // now normalize each sample
    cout << "\nnormalizing all the dpca_samples ...";
    for (unsigned long i = 0; i < dpca_samples.size(); ++i)
        dpca_samples[i] = normalizer(dpca_samples[i]);   
   
    cout << "\n\tdpca_samples size: " << dpca_samples[0].nr()
         << "x" << dpca_samples.size();   
    cout << "\n\tlabels size: " << labels.size() << endl;


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
    cout << "\nrandomizing all the dpca_samples ...";
    randomize_samples(dpca_samples, labels);
    cout << "\n\tdpca_samples size: " << dpca_samples[0].nr()
         << "x" << dpca_samples.size();
    cout << "\n\tlabels size: " << labels.size() << endl;


    // here we make an instance of the svm_nu_trainer object
    svm_nu_trainer<kernel_type> trainer;

    const double max_nu = maximum_nu(labels);

    

    // From looking at the output of the above loop it turns out that good
    // values for C and gamma for this problem are 5 and 0.15625 respectively.
    // So that is what we will use.

    // Now we train on the full set of data and obtain the resulting decision
    // function.  The decision function will return values >= 0 for samples it
    // predicts are in the +1 class and numbers < 0 for samples it predicts to
    // be in the -1 class.
    // doing cross validation ...
/*
    cout << "doing cross validation" << endl;
    for (double gamma = 0.00001; gamma <= 1; gamma *= 5)
    {
        for (double nu = 0.00001; nu < max_nu; nu *= 5)
        {
            // tell the trainer the parameters we want to use
            trainer.set_kernel(kernel_type(gamma));
            trainer.set_nu(nu);

            cout << "gamma: " << gamma << "    nu: " << nu;
            // Print out the cross validation accuracy for 3-fold cross validation using
            // the current gamma and nu.  cross_validate_trainer() returns a row vector.
            // The first element of the vector is the fraction of +1 training examples
            // correctly classified and the second number is the fraction of -1 training
            // examples correctly classified.
            cout << "     cross validation accuracy: " << cross_validate_trainer(trainer, dpca_samples, labels, 3);
        }
    }

*/
 
    trainer.set_kernel(kernel_type(0.00025));
    trainer.set_nu(0.00625);
    
    typedef decision_function<kernel_type> dec_funct_type;
    typedef normalized_function<dec_funct_type> dfunct_type;

    cout << "\nd_svm training start !!!";

    dfunct_type learned_dfunct; 
    learned_dfunct.normalizer = normalizer;
    learned_dfunct.function = trainer.train(dpca_samples, labels);

    cout << "\n\tnumber of support vectors in our learned_dfunct is: "
         << learned_dfunct.function.basis_vectors.size();
    serialize("saved_dfunction.dat") << learned_dfunct;
    cout << "\n\td_svm training complete !!!" << endl;

    // accurary of training set
    get_accurary_cross_training_set(detector, sp, train_positive_dirs, learned_dfunct, +1);

    get_accurary_cross_training_set(detector, sp, train_negative_dirs, learned_dfunct, -1);
    get_accurary_cross_training_set(detector, sp, test_positive_dirs, learned_dfunct, +1);
    get_accurary_cross_training_set(detector, sp, test_negative_dirs, learned_dfunct, -1);
 

    // We can also train a decision function that reports a well conditioned
    // probability instead of just a number > 0 for the +1 class and < 0 for the
    // -1 class.  An example of doing that follows:
    typedef probabilistic_decision_function<kernel_type> probabilistic_funct_type;  
    typedef normalized_function<probabilistic_funct_type> pfunct_type;


    cout << "\np_svm training start !!!";

    pfunct_type learned_pfunct; 
    learned_pfunct.normalizer = normalizer;
    learned_pfunct.function = train_probabilistic_decision_function(trainer, dpca_samples, labels, 3);
    
    
    cout << "\n\tnumber of support vectors in our learned_pfunct is: "
         << learned_pfunct.function.decision_funct.basis_vectors.size();
    
    // Now we have a function that returns the probability that a given sample is of the +1 class.

    // Another thing that is worth knowing is that just about everything in dlib
    // is serializable.  So for example, you can save the learned_pfunct object
    // to disk and recall it later like so:
    serialize("saved_pfunction.dat") << learned_pfunct;
    cout << "\n\tp_svm training complete !!!" << endl;


    // accurary of training set
    get_accurary_cross_training_set(detector, sp, train_positive_dirs, learned_pfunct, +1);

    get_accurary_cross_training_set(detector, sp, train_negative_dirs, learned_pfunct, -1);
    get_accurary_cross_training_set(detector, sp, test_positive_dirs, learned_pfunct, +1);
    get_accurary_cross_training_set(detector, sp, test_negative_dirs, learned_pfunct, -1);
 
}

