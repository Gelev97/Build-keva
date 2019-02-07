/***********************************************************************/

#include "iostream"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

/***********************************************************************/

using namespace std;
using namespace cv;

/***********************************************************************/

#define N_LEVELS 5 // number of scaled images, including original
#define SCALE_FACTOR (1<<N_LEVELS) // ultimate scale factor

// scaled color images
vector<Mat> scaled_images; // vector of scaled images
int sizes[ N_LEVELS ][2]; // sizes of scaled images
int locations[ N_LEVELS ][2]; // locations of images in the display
Mat levels; // display of all color scaled images (3D uchar)

// Variances
Mat sum1[ N_LEVELS ]; // float sum of points so far
Mat sum2[ N_LEVELS ]; // float sum of squares of points so far.
Mat point_count[ N_LEVELS ]; // count of points summed up.
Mat variances[ N_LEVELS ];
Mat levels_var; // display of variances (1D uchar)

Mat blobs[ N_LEVELS ];
Mat levels_blobs; // display of blobs (3D uchar)

// float var_thresholds[ N_LEVELS ] = { 200.0, 100.0, 500.0, 1000.0, 1000.0 };
float var_thresholds[ N_LEVELS ] = { 100.0, 20.0, 100.0, 200.0, 500.0 };

float scale_factor = 1.0;

/***********************************************************************/
/***********************************************************************/
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

  if ( n_rows < n_cols )
    for ( scale_factor = 1; scale_factor < 10; scale_factor += 1 )
      {
	if ( (n_cols/scale_factor <= 1024) && (n_rows/scale_factor <= 768) )
	  break;
      }
  else
    for ( scale_factor = 1; scale_factor < 10; scale_factor += 1 )
      {
	if ( (n_cols/scale_factor <= 768) && (n_rows/scale_factor <= 1024) )
	  break;
      }
    
  printf( "1/scale_factor = %g\n", 1.0/scale_factor );

  // create color display and put original image into them
  levels = Mat::zeros( n_rows + (n_rows >> 1), n_cols, CV_8UC3 );
  printf( "levels: %d %d\n", levels.cols, levels.rows );
  src.copyTo( levels( Rect( 0, 0, src.cols, src.rows ) ));

  // src_copy is a copy of the original image that is the correct size
  Mat src_copy = Mat::zeros( n_rows, n_cols, CV_8UC3 );
  printf( "src_copy: %d %d\n", src_copy.cols, src_copy.rows );
  src.copyTo( src_copy( Rect( 0, 0, src.cols, src.rows ) ));
  // initialize hierarchy of sum arrays
  src_copy.convertTo( sum1[0], CV_32FC3, 1.0 );  
  sum2[0] = Mat::zeros( n_rows, n_cols, CV_32FC3 );
  point_count[0] = Mat::ones( n_rows, n_cols, CV_32FC3 );
  
  // Build an image pyramid and save it in a vector of type Mat
  buildPyramid( src_copy, scaled_images, N_LEVELS-1, BORDER_REPLICATE );

  // print out the sizes of the matrices for debugging
  int i = 0;
  printf( "%d: %d %d; %d %d\n", 0,
	  locations[i][0], locations[i][1],
	  scaled_images[i].cols, scaled_images[i].rows );

  int i_col = 0;
  int i_row = n_rows;
  int nn_cols = n_cols >> 1; 
  int nn_rows = n_rows >> 1;
  for ( int level = 1; level < N_LEVELS; level++ )
    {
      // print out the sizes of the matrices for debugging
      printf( "%d: %d %d; %d %d\n", level, i_col, i_row, nn_cols, nn_rows );
      // hack to set sub-image locations in overall picture
      locations[level][0] = i_col;
      locations[level][1] = i_row;
      // copy the scaled versions of the source image into the display
      scaled_images[ level ].copyTo( levels( Rect( locations[level][0],
					       locations[level][1],
					       scaled_images[level].cols,
					       scaled_images[level].rows ) ));
      // move to the next smaller image
      i_col += scaled_images[ level ].cols;
      nn_cols >>= 1;
      nn_rows >>= 1; 
    }
  
  // show the scaled images
  Mat levels_scaled;
  resize( levels, levels_scaled, Size(), 1/scale_factor, 1/scale_factor );
  if ( n_rows > n_cols )
    rotate( levels_scaled, levels_scaled, ROTATE_90_COUNTERCLOCKWISE );
  imshow( pyramid_window_name, levels_scaled );

  const char* variance_pyramid_window_name = "variance pyramid";

  // Variance matrix displays
  variances[0] = Mat::zeros( n_rows + (n_rows >> 1), n_cols, CV_32F );
  levels_var = Mat::zeros( n_rows + (n_rows >> 1), n_cols, CV_8U );

  // do a special variance computation for level 0, which looks at a 3x3 region
  // go through the pixels
  for ( int r = 0; r < n_rows; r++ )
    {
      for ( int c = 0; c < n_cols; c++ )
	{
	  float varf = 0;
	  for ( int j = 0; j < 3; j++ ) // iterate through colors
	    {
	      // initialize sum2 and point_count
	      float vv = src_copy.at<Vec3b>(r,c)[j];
	      sum2[0].at<Vec3f>(r,c)[j] = vv*vv;
	      point_count[0].at<Vec3f>(r,c)[j] = 1;

	      float v_sum = 0.0;
	      float v_sum2 = 0.0;
	      int n = 0;
	      for ( int dr = -1; dr < 2; dr++ )
		{
		  int ir = r + dr;
		  if ( ir < 0 )
		    continue;
		  if ( ir >= n_rows )
		    continue;
		  for ( int dc = -1; dc < 2; dc++ )
		    {
		      int ic = c + dc;
		      if ( ic < 0 )
			continue;
		      if ( ic >= n_cols )
			continue;
		      int v = src_copy.at<Vec3b>(ir,ic)[j];
		      // sum up the values and the values squared.
		      v_sum += v; // used to calculate the mean
		      v_sum2 += v*v; // used to calculate the variance
		      n++;
		      /*
		      if ( r == 0 && c == 1 )
			printf( "      %d %d %g %g %g %d %d\n",
				ir, ic, varf, v_sum, v_sum2, v, n );
		      */
		    }
		}
	      v_sum /= n;
	      v_sum2 /= n;
	      varf += (v_sum2 - v_sum*v_sum);
	      /*
	      if ( r == 0 && c == 1 )
		printf( "%d %d %g %g %g %d\n", r, c, varf, v_sum, v_sum2, n );
	      */
	    }
	  variances[0].at<float>( r, c ) = varf;
	  if ( varf < 0.0 )
	    {
	      fprintf( stdout, "varf less than zero: %g\n", varf );
	      varf = 0.0;
	    }
	  if ( varf > 255.49 )
	    varf = 255;
	  int vari = round( varf );
	  levels_var.at<uchar>( r, c ) = vari;
	}
      // printf( "%d\n", r );
    }

  printf( "************************\n" );

  // now create the smaller scale versions
  for ( int level = 1; level < N_LEVELS; level++ )
    {
      // create the matrices
      sum1[level] =
	Mat::zeros( scaled_images[level].rows, scaled_images[level].cols,
		    CV_32FC3 );
      sum2[level] =
	Mat::zeros( scaled_images[level].rows, scaled_images[level].cols,
		    CV_32FC3 );
      point_count[level] =
	Mat::zeros( scaled_images[level].rows, scaled_images[level].cols,
		    CV_32FC3 );
      variances[level] = Mat::zeros(
	    scaled_images[level].rows, scaled_images[level].cols, CV_32F );
      // go through the pixels
      for ( int r = 0; r < scaled_images[level].rows; r++ )
	{
	  for ( int c = 0; c < scaled_images[level].cols; c++ )
	    {
	      float varf = 0;
	      for ( int j = 0; j < 3; j++ ) // iterate through colors
		{
		  float v_sum = 0.0;
		  float v_sum2 = 0.0;
		  float v_count = 0.0;
		  for ( int dr = 0; dr < 2; dr++ )
		    {
		      int ir = 2*r + dr;
		      for ( int dc = 0; dc < 2; dc++ )
			{
			  int ic = 2*c + dc;
			  float v = sum1[level-1].at<Vec3f>(ir,ic)[j];
			  // sum up the values and the values squared.
			  v_sum += v; // used to calculate the mean
			  // v_sum2 += v*v; // used to calculate the variance
			  v_sum2 += sum2[level-1].at<Vec3f>(ir,ic)[j];
			  v_count += point_count[level-1].at<Vec3f>(ir,ic)[j];
			  /*
			  if ( r == 0 && c == 1 )
			    printf( "      %d %d %g %g %g %f %f\n",
				    ir, ic, varf, v_sum, v_sum2, v, v_count );
			  */
			}
		    }
		  sum1[level].at<Vec3f>(r,c)[j] = v_sum;
		  sum2[level].at<Vec3f>(r,c)[j] = v_sum2;
		  point_count[level].at<Vec3f>(r,c)[j] = v_count;
		  varf += v_sum2/v_count - v_sum*v_sum/(v_count*v_count);
		  /*
		  if ( r == 0 && c == 1 )
		    printf( "%d %d %d %g %g %g %f\n",
			    level, r, c, varf, v_sum, v_sum2, v_count );
		  */
		}
	      variances[level].at<float>( r, c ) = varf;
	      if ( varf < 0.0 )
		{
		  fprintf( stdout, "varf less than zero: %g\n", varf );
		  varf = 0.0;
		}
	      if ( varf > 255.49 )
		varf = 255;
	      int vari = round( varf );
	      levels_var.at<uchar>
		( r + locations[level][1], c + locations[level][0] )
		= vari;
	      /*
	      if ( r==0 && c==0 )
	       printf( "%d %d; %d %d; %d %d; %d\n",
			locations[level][0], locations[level][1],
			r, c,
			r + locations[level][0], c + locations[level][1],
			vari );
	      */
	    }
	}
    }

  // show the displays
  resize( levels_var, levels_scaled, Size(), 1/scale_factor, 1/scale_factor );
  if ( n_rows > n_cols )
    rotate( levels_scaled, levels_scaled, ROTATE_90_COUNTERCLOCKWISE );
  imshow( variance_pyramid_window_name, levels_scaled );

  // Blob displays
  const char* blobs_pyramid_window_name = "blobs pyramid";
  levels_blobs = Mat::zeros( n_rows + (n_rows >> 1), n_cols, CV_8UC3 );

  // do a special blob computation for level 0, which looks at a 3x3 region
  // go through the pixels
  for ( int r = 0; r < n_rows; r++ )
    {
      for ( int c = 0; c < n_cols; c++ )
	{
	  float varf = 0.0;
	  for ( int dr = -2; dr <= 2; dr++ )
	    {
	      int ir = r + dr;
	      if ( ir < 0 )
		continue;
	      if ( ir >= n_rows )
		continue;
	      for ( int dc = -2; dc <= 2; dc++ )
		{
		  int ic = c + dc;
		  if ( ic < 0 )
		    continue;
		  if ( ic >= n_cols )
		    continue;
		  varf += variances[0].at<float>( r, c );
		}
	    }
	  if ( varf < var_thresholds[ 0 ] )
	    levels_blobs.at<Vec3b>( r, c )[2] = 255;
	}
    }

  printf( "************************\n" );

  // now create the smaller scale versions
  for ( int level = 1; level < N_LEVELS; level++ )
    {
      // go through the pixels
      for ( int r = 0; r < scaled_images[level].rows; r++ )
	{
	  for ( int c = 0; c < scaled_images[level].cols; c++ )
	    {
	      float varf = 0;
	      for ( int dr = -2; dr <= 2; dr++ )
		{
		  int ir = r + dr;
		  if ( ir < 0 )
		    continue;
		  if ( ir >= n_rows )
		    continue;
		  for ( int dc = -2; dc <= 2; dc++ )
		    {
		      int ic = c + dc;
		      if ( ic < 0 )
			continue;
		      if ( ic >= n_cols )
			continue;
		      varf += variances[level].at<float>( r, c );
		    }
		}
	      // printf( "%d %d %g\n", r, c, varf );
	      if ( varf < var_thresholds[ level ] )
		levels_blobs.at<Vec3b>
		  ( r + locations[level][1], c + locations[level][0] )[2]
		  = 255;
	    }
	}
    }

  // show the displays
  resize( levels_blobs, levels_scaled, Size(),
	  1/scale_factor, 1/scale_factor );
  if ( n_rows > n_cols )
    rotate( levels_scaled, levels_scaled, ROTATE_90_COUNTERCLOCKWISE );
  imshow( blobs_pyramid_window_name, levels_scaled );

  waitKey(0);

  return 0;
}

/***********************************************************************/
