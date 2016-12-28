#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <dlib/image_processing.h>

#include <dlib/matrix.h>
using namespace dlib;
using namespace std;



// ANISODIFF - Anisotropic diffusion.
//
// Usage: 
// void = anisodiff(im, diff, niter, kappa, lambda, option)
// Arguments:
//      im      - input image
//      niter   - number of iterations.
//      kappa   - conduction coefficient 20-100 ?
//      lambda  - max valiue of .25 for stability
//      option  - 1 Perona Malik diffusion equation No 1
//                2 Perona Malik diffusion equation No 2
// Returns:
//      diff    - diffused image.

// kappa controls conduction as a function of gradient.  If kappa is low
// small intensity gradients are able to block conduction and hence diffusion
// across step edges.  A large value reduces the influence of intensity
// gradients on conduction.
//
// lambda controls speed of diffusion (you usually want it at a maximum of
// 0.25)
//
// Diffusion equation 1 favours high contrast edges over low contrast ones.
// Diffusion equation 2 favours wide regions over smaller ones.
//
template<
    typename image_type
    >
void anisodiff(
        const image_type& im,
        image_type& diff,
        const unsigned long niter, 
        const int kappa, 
        const double lambda, 
        const int option
)
{
    // Reference - http://blog.csdn.net/bluecol/article/details/46690985
    
    dlib::assign_image(diff, im);

    for (unsigned long i = 0; i < niter; i++)
    {
        image_type diffl(diff.nr()+2, diff.nc()+2);
        assign_all_pixels(diffl, 0);

        for (unsigned long r = 0; r < diffl.nr(); r++)
        {
            for (unsigned long c = 0; c < diffl.nc(); c++)
            {
                if (r == 0 || c == 0 || r == diff.nr() || c == diff.nc())
                    assign_pixel(diffl[r][c], 0);
                else
                    assign_pixel(diffl[r][c], diff[r-1][c-1]);
            }
        }

        //deltaN
        image_type deltaN(diff.nr(), diff.nc());
        dlib::assign_image(deltaN, 
            (subm(mat(diffl), range(0, diff.nr()-1), range(1, diff.nc())) - mat(diff)));

        //deltaS
        image_type deltaS(diff.nr(), diff.nc());
        dlib::assign_image(deltaS, 
            (subm(mat(diffl), range(2, diff.nr()+1), range(1, diff.nc())) - mat(diff)));

        //deltaE
        image_type deltaE(diff.nr(), diff.nc());
        dlib::assign_image(deltaE, 
            (subm(mat(diffl), range(1, diff.nr()), range(2, diff.nc()+1)) - mat(diff)));

        //deltaW
        image_type deltaW(diff.nr(), diff.nc());
        dlib::assign_image(deltaW, 
            (subm(mat(diffl), range(1, diff.nr()), range(0, diff.nc()-1)) - mat(diff)));

        // Conduction
        image_type cN(diff.nr(), diff.nc());
        image_type cS(diff.nr(), diff.nc());
        image_type cE(diff.nr(), diff.nc());    
        image_type cW(diff.nr(), diff.nc());
        
        if (option == 1)
        { 
            dlib::assign_image(cN, -1 * pointwise_multiply( mat(deltaN)/kappa , mat(deltaN)/kappa ));
            dlib::assign_image(cS, -1 * pointwise_multiply( mat(deltaS)/kappa , mat(deltaS)/kappa ));
            dlib::assign_image(cE, -1 * pointwise_multiply( mat(deltaE)/kappa , mat(deltaE)/kappa ));
            dlib::assign_image(cW, -1 * pointwise_multiply( mat(deltaW)/kappa , mat(deltaW)/kappa ));
            for (unsigned long r = 0; r < diff.nr(); r++)
            {
                for (unsigned long c = 0; c < diff.nc(); c++)
                {
                    dlib::assign_pixel(cN[r][c], exp(cN[r][c]));
                    dlib::assign_pixel(cS[r][c], exp(cS[r][c]));
                    dlib::assign_pixel(cE[r][c], exp(cE[r][c]));
                    dlib::assign_pixel(cW[r][c], exp(cW[r][c]));
                }
            }
        }
        else if (option == 2)
        {
            dlib::assign_image(cN, 1 / ( 1 + pointwise_multiply( mat(deltaN)/kappa , mat(deltaN)/kappa )));
            dlib::assign_image(cS, 1 / ( 1 + pointwise_multiply( mat(deltaS)/kappa , mat(deltaS)/kappa )));
            dlib::assign_image(cE, 1 / ( 1 + pointwise_multiply( mat(deltaE)/kappa , mat(deltaE)/kappa )));
            dlib::assign_image(cW, 1 / ( 1 + pointwise_multiply( mat(deltaW)/kappa , mat(deltaW)/kappa )));
        }

        image_type conduction(diff.nr(), diff.nc());
        dlib::assign_image(conduction, 
            pointwise_multiply(mat(cN), mat(deltaN))
            + pointwise_multiply(mat(cS), mat(deltaS))
            + pointwise_multiply(mat(cE), mat(deltaE))
            + pointwise_multiply(mat(cW), mat(deltaW)));

        dlib::assign_image(diff, mat(diff) + lambda * mat(conduction));
    }

}
//---------------------------------------------------------------------------------------------------------

// EXTRACT FEATURES - Extract customize lbp desscriptors
//
// Usage: 
// void = extract_customize_lbp_descriptors(img, det, feats, num_scales)
// Arguments:
//      img     - input image
//      det     - the object information of input
//      feats   - store features form descriptor
//      num_scales  - the scale of image size (>=0) to extract features. Default is 5.
// Returns:
//      feats   - diffused image.
//
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

    array2d<rgb_pixel> img_fixed(120, 120);
    resize_image(img, img_fixed);

    array2d<unsigned char> lbp;
    make_uniform_lbp_image(img_fixed, lbp);

    std::vector<point> parts;
    parts.push_back(det.part(30));

    for (unsigned long i = 0; i < parts.size(); ++i)
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

            for (unsigned long i = 0; i < parts.size(); ++i)
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
