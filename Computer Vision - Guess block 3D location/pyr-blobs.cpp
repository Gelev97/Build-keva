/***********************************************************************/

#include "iostream"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

/***********************************************************************/

using namespace std;
using namespace cv;

/***********************************************************************/

#define N_LEVELS 6 // number of scaled images, including original
#define SCALE_FACTOR (1<<N_LEVELS) // ultimate scale factor

// scaled color images
vector<Mat> scaled_images; // vector of scaled images
int sizes[ N_LEVELS ][2]; // sizes of scaled images
int locations[ N_LEVELS ][2]; // locations of images in the display
Mat levels; // display of all color scaled images

// B/W averaged images
Mat levels_bw; // display of all b/w scaled images
Mat means_bwf[ N_LEVELS ]; // b/w float means of pixels so far
Mat means_bw[ N_LEVELS ]; // b/w uchar means of pixels so far

// Variances
Mat sum2_bwf[ N_LEVELS ]; // b/w float mean sum2 of pixels so far
Mat levels_var_bw; // display of all b/w sum2
Mat variances_bwf[ N_LEVELS ]; // b/w variances of pixels so far
Mat variances_bw[ N_LEVELS ]; // b/w variances of pixels so far
Mat levels_scaled_var_bw; // display of all b/w sum2
Mat scaled_variances_bwf[ N_LEVELS ]; // b/w scaled variances of pixels so far
Mat scaled_variances_bw[ N_LEVELS ]; // b/w scaled variances of pixels so far

/***********************************************************************/

int main( int argc, char** argv )
{
  const char* pyramid_window_name = "Image pyramid";
  const char* filename = argc >=2 ? argv[1] : "planks1ss.jpg";

  // Get an image
  Mat src = imread( filename );
  // Check if image is loaded fine
  if( src.empty() )
    {
      printf(" Error opening image\n");
      printf(" Program Arguments: [image_name]\n");
      return -1;
    }

  // figure out what image sizes we are going to actually use
  // round size up so can do N_LEVEL-1 shrinks
  int n_cols = ((src.cols + SCALE_FACTOR - 1) >> N_LEVELS) << N_LEVELS;
  int n_rows = ((src.rows + SCALE_FACTOR - 1) >> N_LEVELS) << N_LEVELS;
  printf( "%d %d; %d %d; %d %d\n", N_LEVELS, SCALE_FACTOR, src.cols, src.rows,
	  n_cols, n_rows );

  // create color display and put original image into them
  levels = Mat::zeros( n_rows + (n_rows >> 1), n_cols, CV_8UC3 );
  printf( "levels: %d %d\n", levels.cols, levels.rows );
  src.copyTo( levels( Rect( 0, 0, src.cols, src.rows ) ));

  // src_copy is a copy of the original image that is the correct size
  Mat src_copy = Mat::zeros( n_rows, n_cols, CV_8UC3 );
  printf( "src_copy: %d %d\n", src_copy.cols, src_copy.rows );
  src.copyTo( src_copy( Rect( 0, 0, src.cols, src.rows ) ));
  
  // Build an image pyramid and save it in a vector of type Mat
  buildPyramid( src_copy, scaled_images, N_LEVELS-1, BORDER_REPLICATE );

  // print out the sizes of the matrices for debugging
  int i = 0;
  printf( "%d: %d %d; %d %d\n", 0,
	  locations[i][0], locations[i][1],
	  scaled_images[i].cols, scaled_images[i].rows );

  int i_col = 0;
  int i_row = n_rows;
  n_cols >>= 1; 
  n_rows >>= 1;
  for ( int i = 1; i < N_LEVELS; i++ )
    {
      // print out the sizes of the matrices for debugging
      printf( "%d: %d %d; %d %d\n", i, i_col, i_row, n_cols, n_rows );
      // hack to set sub-image locations in overall picture
      locations[i][0] = i_col;
      locations[i][1] = i_row;
      // copy the scaled versions of the source image into the display
      scaled_images[ i ].copyTo( levels( Rect( locations[i][0],
					       locations[i][1],
					       scaled_images[i].cols,
					       scaled_images[i].rows ) ));
      // move to the next smaller image
      i_col += scaled_images[ i ].cols;
      n_cols >>= 1;
      n_rows >>= 1; 
    }
  
  // show the scaled images
  imshow( pyramid_window_name, levels );

  // create black and white image pyramid
  n_cols = ((src.cols + SCALE_FACTOR - 1) >> N_LEVELS) << N_LEVELS;
  n_rows = ((src.rows + SCALE_FACTOR - 1) >> N_LEVELS) << N_LEVELS;

  const char* bw_pyramid_window_name = "B/W image pyramid";
  const char* bw2_pyramid_window_name = "variance pyramid";
  const char* bw2s_pyramid_window_name = "scaled variance pyramid";
  Mat src_gray; // B/W version of source image
  cvtColor( src, src_gray, COLOR_BGR2GRAY );
  // printf( "src_gray: %d %d\n", src_gray.cols, src_gray.rows );

  // create B/W display and put original B/W image into them
  levels_bw = Mat::zeros( n_rows + (n_rows >> 1), n_cols, CV_8U );
  // printf( "levels_bw: %d %d\n", levels_bw.cols, levels_bw.rows );
  src_gray.copyTo( levels_bw( Rect( 0, 0, src_gray.cols, src_gray.rows ) ));

  // Variance displays
  levels_var_bw = Mat::zeros( n_rows + (n_rows >> 1), n_cols, CV_8U );
  levels_scaled_var_bw = Mat::zeros( n_rows + (n_rows >> 1), n_cols, CV_8U );

  // largest scale image of means
  means_bw[0] = Mat::zeros( n_rows, n_cols, CV_8U );
  src_gray.copyTo( means_bw[0]( Rect( 0, 0, src_gray.cols, src_gray.rows ) ));
  means_bw[0].copyTo( levels_bw( Rect( 0, 0,
				       means_bw[0].cols, means_bw[0].rows ) ));

  // set up level 0 of float means
  means_bw[0].convertTo( means_bwf[0], CV_32F, 1.0 );  

  // set up level 0 of float average sum2
  sum2_bwf[0] = Mat::zeros( scaled_images[0].rows, scaled_images[0].cols,
			    CV_32F );

  // initialize the largest scale sum of the means squared.
  for ( int r = 0; r < means_bw[0].rows; r++ )
    {
      for ( int c = 0; c < means_bw[0].cols; c++ )
	{
	  int v = means_bw[0].at<uchar>(r,c);
	  sum2_bwf[0].at<float>(r,c) = v*v;
	}
    }

  // initialize the variances at level 0
  variances_bwf[0] = Mat::zeros( scaled_images[0].rows, scaled_images[0].cols,
				 CV_32F );
  variances_bw[0] = Mat::zeros( scaled_images[0].rows, scaled_images[0].cols,
				CV_8U );
  scaled_variances_bwf[0] = Mat::zeros( scaled_images[0].rows,
					scaled_images[0].cols, CV_32F );
  scaled_variances_bw[0] = Mat::zeros( scaled_images[0].rows,
				       scaled_images[0].cols, CV_8U );

  // now create the smaller scale versions
  for ( int i = 1; i < N_LEVELS; i++ )
    {
      // create the matrices
      means_bwf[i] = Mat::zeros( scaled_images[i].rows, scaled_images[i].cols,
				CV_32F );
      sum2_bwf[i] = Mat::zeros( scaled_images[i].rows, scaled_images[i].cols,
				CV_32F );
      variances_bwf[i] = Mat::zeros( scaled_images[i].rows,
				     scaled_images[i].cols,
				     CV_32F );
      scaled_variances_bwf[i] = Mat::zeros( scaled_images[i].rows,
					    scaled_images[i].cols,
					    CV_32F );
      /*
      printf( "means_bwf[ %d ]: %d %d; %d %d\n", i,
	      means_bwf[i].cols, means_bwf[i].rows,
	      scaled_images[i].cols, scaled_images[i].rows );
      */
      // go through the pixels
      for ( int r = 0; r < scaled_images[i].rows; r++ )
	{
	  for ( int c = 0; c < scaled_images[i].cols; c++ )
	    {
	      float sum = 0.0;
	      float sum2 = 0.0;
	      for ( int dr = 0; dr < 2; dr++ )
		{
		  int ir = 2*r + dr;
		  for ( int dc = 0; dc < 2; dc++ )
		    {
		      int ic = 2*c + dc;
		      float v = means_bwf[i-1].at<float>(ir,ic);
		      // sum up the values and the values squared.
		      sum += v; // used to calculate the mean
		      sum2 += v*v; // used to calculate the variance
		    }
		}
	      sum /= 4; // using a 2x2 window with 4 elements
	      sum2 /= 4;
	      means_bwf[i].at<float>(r,c) = sum;
	      sum2_bwf[i].at<float>(r,c) = sum2;
	      variances_bwf[i].at<float>(r,c) = sum2 - sum*sum;
	      // scale the variances by the mean^2
	      if ( sum > 0.0 )
		scaled_variances_bwf[i].at<float>(r,c) =
		  100000*variances_bwf[i].at<float>(r,c)/
		  (means_bwf[i].at<float>(r,c)*
		   means_bwf[i].at<float>(r,c));
	      else
		scaled_variances_bwf[i].at<float>(r,c) = 1e7;
	    }
	}
      // convert from floats to ints for display
      means_bwf[i].convertTo( means_bw[i], CV_8U, 1.0 );  
      means_bw[i].copyTo( levels_bw( Rect( locations[i][0],
					   locations[i][1],
					   scaled_images[i].cols,
					   scaled_images[i].rows ) ));
      variances_bwf[i].convertTo( variances_bw[i], CV_8U, 1.0 );  
      variances_bw[i].copyTo( levels_var_bw( Rect( locations[i][0],
						   locations[i][1],
						   scaled_images[i].cols,
						   scaled_images[i].rows ) ));
      scaled_variances_bwf[i].convertTo(
		scaled_variances_bw[i], CV_8U, 1.0 );  
      scaled_variances_bw[i].copyTo(
				    levels_scaled_var_bw(
				      Rect( locations[i][0],
				            locations[i][1],
					    scaled_images[i].cols,
					    scaled_images[i].rows ) ));
    }

  // show the displays
  imshow( bw_pyramid_window_name, levels_bw );
  imshow( bw2_pyramid_window_name, levels_var_bw );
  imshow( bw2s_pyramid_window_name, levels_scaled_var_bw );

  waitKey(0);

  return 0;
}

/***********************************************************************/
