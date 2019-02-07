/***********************************************************************/

#include "iostream"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

/***********************************************************************/

using namespace std;
using namespace cv;

/***********************************************************************/

#define MAX_N_LEVELS 10 // number of scaled images, including original

/***********************************************************************/

int n_levels = 0; // actual number of levels

// scaled color images
vector<Mat> scaled_images; // vector of scaled images

bool display_this_level[ MAX_N_LEVELS ]; // do we display this level
int sizes[ MAX_N_LEVELS ][2]; // sizes of scaled images
int locations[ MAX_N_LEVELS ][2]; // locations of images in the display
Mat levels; // display of all color scaled images (3D uchar)

// Variances
Mat sum1[ MAX_N_LEVELS ]; // float sum of points so far
Mat sum2[ MAX_N_LEVELS ]; // float sum of squares of points so far.
Mat point_count[ MAX_N_LEVELS ]; // count of points summed up.
Mat variances[ MAX_N_LEVELS ];
Mat levels_var; // display of variances (1D uchar)

Mat blobs[ MAX_N_LEVELS ];
Mat levels_blobs; // display of blobs (3D uchar)

// float var_thresholds[ N_LEVELS ] = { 200.0, 100.0, 500.0, 1000.0, 1000.0 };
float var_thresholds[ MAX_N_LEVELS ] = { 500.0, 750.0, 1000.0, 2000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0 };

/***********************************************************************/
/***********************************************************************/

Mat src_canny, src_gray_canny;
Mat dst_canny, detected_edges_canny;
int lowThreshold = 100;
const int max_lowThreshold = 350;
const int ratio = 3;
const int kernel_size = 3;
const char* canny_window_name = "Canny edges based on variance";

static void CannyThreshold(int, void*)
{
    blur( src_gray_canny, detected_edges_canny, Size(3,3) );
    Canny( detected_edges_canny, detected_edges_canny,
	   lowThreshold, lowThreshold*ratio, kernel_size );
    dst_canny = Scalar::all(0);
    src_canny.copyTo( dst_canny, detected_edges_canny );
    imshow( canny_window_name, dst_canny );
}

/***********************************************************************/

Mat src_canny2, src_gray_canny2;
Mat dst_canny2, detected_edges_canny2;
int lowThreshold2 = 100;
const int max_lowThreshold2 = 350;
const int ratio2 = 3;
const int kernel_size2 = 3;
const char* canny2_window_name = "Canny edges based on image";

static void CannyThreshold2(int, void*)
{
    blur( src_gray_canny2, detected_edges_canny2, Size(3,3) );
    Canny( detected_edges_canny2, detected_edges_canny2,
	   lowThreshold2, lowThreshold2*ratio2, kernel_size2 );
    dst_canny2 = Scalar::all(0);
    src_canny2.copyTo( dst_canny2, detected_edges_canny2 );
    imshow( canny2_window_name, dst_canny2 );
}

/***********************************************************************/

int main( int argc, char** argv )
{
  const char* pyramid_window_name = "Image pyramid";
  const char* filename = argc >=2 ? argv[1] : "planks1.jpg";

  // Get an image
  Mat src = imread( filename );
  // Check if image is loaded fine
  if( src.empty() )
    {
      printf(" Error opening image\n");
      printf(" Program Arguments: [image_name]\n");
      return -1;
    }

  // figure out how many levels of shrinkage we are going to apply
  int r = src.rows;
  int c = src.cols;
  for ( n_levels = 1; n_levels <= MAX_N_LEVELS; n_levels++ )
    {
      printf( "level %d: %d %d\n", n_levels - 1, r, c );
      if ( src.rows < src.cols )
	{
	  if ( r < 32 || c < 64 )
	    break;
	}
      else
	{
	  if ( c < 32 || r < 64 )
	    break;
	}
      r >>= 1;
      c >>= 1;
    }

  int scale_factor = 1 << n_levels;
	  
  // figure out what image sizes we are going to actually use
  // round size up so can do N_LEVEL-1 shrinks
  int n_cols = ((src.cols + scale_factor - 1) >> n_levels) << n_levels;
  int n_rows = ((src.rows + scale_factor - 1) >> n_levels) << n_levels;
  printf( "%d %d; %d %d; %d %d\n", n_levels, scale_factor,
	  src.cols, src.rows, n_cols, n_rows );

  // src_copy is a copy of the original image that is the correct size
  Mat src_copy = Mat::zeros( n_rows, n_cols, CV_8UC3 );
  printf( "src_copy: %d %d\n", src_copy.cols, src_copy.rows );
  src.copyTo( src_copy( Rect( 0, 0, src.cols, src.rows ) ));
  // initialize hierarchy of sum arrays
  src_copy.convertTo( sum1[0], CV_32FC3, 1.0 );  
  sum2[0] = Mat::zeros( n_rows, n_cols, CV_32FC3 );
  point_count[0] = Mat::ones( n_rows, n_cols, CV_32FC3 );
  
  // Build an image pyramid and save it in a vector of type Mat
  buildPyramid( src_copy, scaled_images, n_levels-1, BORDER_REPLICATE );

  // figure out what scaled images we can actually show on our screen
  int level_start = n_levels - 5; 
  if ( level_start < 0 )
    level_start = 0;

  printf( "level_start: %d\n", level_start );
  int level = 0;
  for ( level = 0; level < n_levels; level++ )
    {
      if ( level < level_start )
	display_this_level[ level ] = false;
      else
	display_this_level[ level ] = true;
      locations[ level ][ 0 ] = 0;
      locations[ level ][ 1 ] = 0;
      sizes[ level ][ 0 ] = scaled_images[ level ].cols;
      sizes[ level ][ 1 ] = scaled_images[ level ].rows;
    }
  
  int levels_cols;
  int levels_rows;
  // create color display and put original image into them
  if ( n_cols >= n_rows )
    {
      levels_cols = sizes[ level_start ][ 0 ];
      levels_rows = sizes[ level_start ][ 1 ];
    }
  else
    {
      // transposing the picture
      levels_cols = sizes[ level_start ][ 1 ];
      levels_rows = sizes[ level_start ][ 0 ];
    }
  levels_rows = levels_rows + (levels_rows >> 1);
  levels = Mat::zeros( levels_rows, levels_cols, CV_8UC3 );
  printf( "levels: %d %d\n", levels.cols, levels.rows );

  // print out the sizes of the matrices for debugging
  level = 0;
  printf( "%d: %s %d %d; %d %d\n", level,
	  (display_this_level[ level ]) ? "display" : "not displayed",
	  locations[ level ][0], locations[ level ][1],
	  sizes[ level ][ 0 ], sizes[ level ][ 1 ] );

  // copy the first image into the display
  if ( level_start == 0 )
    {
      if ( n_cols >= n_rows )
	{
	  scaled_images[0].copyTo( levels( Rect( locations[ level_start ][0],
						 locations[ level_start ][1],
						 sizes[ level_start ][0],
						 sizes[ level_start ][1] ) ) );
	}      
      else
	{
	  Mat m;
	  rotate( scaled_images[0], m, ROTATE_90_COUNTERCLOCKWISE );
	  printf( "m: %d %d\n", m.cols, m.rows );
	  m.copyTo( levels_var( Rect( locations[ level_start ][0],
				      locations[ level_start ][1],
				      sizes[ level_start ][1],
				      sizes[ level_start ][0] ) ) );
	}
    }

  int i_col = 0;
  int i_row = 0;  
  if ( n_cols >= n_rows )
    i_row = sizes[ level_start ][ 1 ];
  else
    i_row = sizes[ level_start ][ 0 ];
  for ( level = 1; level < n_levels; level++ )
    {
      // hack to set sub-image locations in overall picture
      if ( level > level_start )
	{
	  locations[level][0] = i_col;
	  locations[level][1] = i_row;
	}
      // print out the sizes of the matrices for debugging
      if ( n_cols >= n_rows )
	{
	  printf( "%d: %s %d %d; %d %d\n", level,
		  (display_this_level[ level ]) ? "display" : "not displayed",
		  locations[ level ][0], locations[ level ][1],
		  sizes[ level ][ 0 ], sizes[ level ][ 1 ] );
	}
      else
	{
	  printf( "%d: %s %d %d; %d %d\n", level,
		  (display_this_level[ level ]) ? "display" : "not displayed",
		  locations[ level ][0], locations[ level ][1],
		  sizes[ level ][ 1 ], sizes[ level ][ 0 ] );
	}
      if ( level >= level_start )
	{
	  if ( n_cols >= n_rows )
	    {
	      // copy the scaled versions of the source image into the display
	      scaled_images[ level ].copyTo( levels( Rect( locations[level][0],
							   locations[level][1],
							   sizes[level][0],
							   sizes[level][1] ) ) );
	    }
	  else
	    {
	      Mat m;
	      rotate( scaled_images[ level ], m, ROTATE_90_COUNTERCLOCKWISE );
	      // copy the scaled versions of the source image into the display
	      m.copyTo( levels( Rect( locations[level][0],
				      locations[level][1],
				      sizes[level][1],
				      sizes[level][0] ) ) );
	    }
	}
      if ( level > level_start )
	{
	  // move to the next smaller image
	  if ( n_cols >= n_rows )
	    i_col += scaled_images[ level ].cols;
	  else
	    i_col += scaled_images[ level ].rows;
	}
    }
  printf( "i_col: %d\n", i_col );
  
  imshow( pyramid_window_name, levels );

  printf( "var 1 *********************\n" );

  const char* variance_pyramid_window_name = "variance pyramid";

  // Variance matrix displays
  variances[0] = Mat::zeros( n_rows, n_cols, CV_32F );
  levels_var = Mat::zeros( levels.rows, levels.cols, CV_8U );

  Mat vari_m = Mat::zeros( n_rows, n_cols, CV_8U );

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
	  vari_m.at<uchar>( r, c ) = vari;
	}
      // printf( "%d\n", r );
    }

  // copy the scaled versions of the variances into the display
  if ( level_start == 0 )
    {
      if ( n_cols >= n_rows )
	{
	  vari_m.copyTo( levels_var( Rect( locations[ level_start ][0],
					   locations[ level_start ][1],
					   sizes[ level_start ][0],
					   sizes[ level_start ][1] ) ) );
	}      
      else
	{
	  Mat m;
	  rotate( vari_m, m, ROTATE_90_COUNTERCLOCKWISE );
	  printf( "m: %d %d\n", m.cols, m.rows );
	  m.copyTo( levels_var( Rect( locations[ level_start ][0],
				      locations[ level_start ][1],
				      sizes[ level_start ][1],
				      sizes[ level_start ][0] ) ) );
	}
    }

  printf( "var 2 *********************\n" );

  // now create the smaller scale versions
  for ( level = 1; level < n_levels; level++ )
    {
      Mat vari_m = Mat::zeros( sizes[ level ][1], sizes[ level ][0], CV_8U );
      printf( "Variance level %d; %d %d; %d %d; %d %d\n",
	      level, vari_m.cols, vari_m.rows,
	      scaled_images[ level ].cols, scaled_images[ level ].rows,
	      sizes[ level ][0], sizes[ level ][1] );
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
	      vari_m.at<uchar>( r, c ) = vari;
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

      // copy the scaled versions of the variances into the display
      if ( level >= level_start )
	{
	  if ( n_cols >= n_rows )
	    {
	      vari_m.copyTo( levels_var( Rect( locations[ level ][0],
					       locations[ level ][1],
					       sizes[ level ][0],
					       sizes[ level ][1] ) ) );
	    }
	  else
	    {
	      Mat m;
	      rotate( vari_m, m, ROTATE_90_COUNTERCLOCKWISE );
	      m.copyTo( levels_var( Rect( locations[ level ][0],
					  locations[ level ][1],
					  sizes[ level ][1],
					  sizes[ level ][0] ) ) );
	    }
	}
    }

  // show the variances
  imshow( variance_pyramid_window_name, levels_var );

  printf( "blobs 1 *********************\n" );

  // Blob displays
  const char* blobs_pyramid_window_name = "blobs pyramid";
  levels_blobs = Mat::zeros( levels.rows, levels.cols, CV_8UC3 );
  printf( "levels_blobs: %d %d\n", levels_blobs.cols, levels_blobs.rows );

  Mat blobs_m0 = Mat::zeros( n_rows, n_cols, CV_8UC3 );

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
	  if ( varf < var_thresholds[0] )
	    {
	      blobs_m0.at<Vec3b>( r, c )[0] = 255;
	      blobs_m0.at<Vec3b>( r, c )[1] = 255;
	      blobs_m0.at<Vec3b>( r, c )[2] = 255;
	    }
	}
    }

  // copy the scaled versions of the variances into the display
  if ( level_start == 0 )
    {
      if ( n_cols >= n_rows )
	{
	  blobs_m0.copyTo( levels_blobs( Rect( locations[ level_start ][0],
					       locations[ level_start ][1],
					       sizes[ level_start ][0],
					       sizes[ level_start ][1] ) ) );
	}      
      else
	{
	  Mat m;
	  rotate( blobs_m0, m, ROTATE_90_COUNTERCLOCKWISE );
	  printf( "m: %d %d\n", m.cols, m.rows );
	  m.copyTo( levels_blobs( Rect( locations[ level_start ][0],
					locations[ level_start ][1],
					sizes[ level_start ][1],
					sizes[ level_start ][0] ) ) );
	}
    }

  printf( "blobs 2 ********************\n" );

  level = 1;
  printf( "Blobs level %d; %d %d; %d %d\n",
	  level, scaled_images[ level ].cols, scaled_images[ level ].rows,
	  sizes[ level ][0], sizes[ level ][1] );

  // now create the smaller scale versions
  for ( level = 1; level < n_levels; level++ )
    {
      Mat blobs_m = Mat::zeros( sizes[ level ][1], sizes[ level ][0],
				CV_8UC3 );
      printf( "Blobs level %d; %d %d; %d %d; %d %d\n",
	      level, blobs_m.cols, blobs_m.rows,
	      scaled_images[ level ].cols, scaled_images[ level ].rows,
	      sizes[ level ][0], sizes[ level ][1] );
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
	      if ( varf < var_thresholds[level] )
		{
		  blobs_m.at<Vec3b>( r, c )[0] = 255;
		  blobs_m.at<Vec3b>( r, c )[1] = 255;
		  blobs_m.at<Vec3b>( r, c )[2] = 255;
		}
	    }
	}

      // copy the scaled versions of the variances into the display
      if ( level >= level_start )
	{
	  if ( n_cols >= n_rows )
	    {
	      blobs_m.copyTo( levels_blobs( Rect( locations[ level ][0],
						  locations[ level ][1],
						  sizes[ level ][0],
						  sizes[ level ][1] ) ) );
	    }
	  else
	    {
	      Mat m;
	      rotate( blobs_m, m, ROTATE_90_COUNTERCLOCKWISE );
	      m.copyTo( levels_blobs( Rect( locations[ level ][0],
					    locations[ level ][1],
					    sizes[ level ][1],
					    sizes[ level ][0] ) ) );
	    }
	}
    }
  
  // show the blobs
  imshow( blobs_pyramid_window_name, levels_blobs );

  bool useRefine = false;

  printf( "LSD lines based on variances ****************\n" );

  Mat levels_gray;
  cvtColor( levels, levels_gray, COLOR_BGR2GRAY );

  // Create and LSD detector with standard or no refinement.
  Ptr<LineSegmentDetector> ls = useRefine ?
    createLineSegmentDetector(LSD_REFINE_STD) :
    createLineSegmentDetector(LSD_REFINE_NONE);

  vector<Vec4f> lines_std;
  
  // Detect the lines
  ls->detect( levels_gray, lines_std );

  levels_gray = Scalar(0, 0, 0);

  ls->drawSegments( levels_gray, lines_std );

  // show the lines
  const char* lines_pyramid_window_name = "LSD lines based on variances";
  imshow( lines_pyramid_window_name, levels_gray );

  printf( "LSD lines based on blobs ****************\n" );

  Mat levels_gray2;
  cvtColor( levels_blobs, levels_gray2, COLOR_BGR2GRAY );

  // Create and LSD detector with standard or no refinement.
  Ptr<LineSegmentDetector> ls2 = useRefine ?
    createLineSegmentDetector(LSD_REFINE_STD) :
    createLineSegmentDetector(LSD_REFINE_NONE);

  vector<Vec4f> lines_std2;
  
  // Detect the lines
  ls2->detect( levels_gray2, lines_std2 );

  levels_gray2 = Scalar(0, 0, 0);

  ls2->drawSegments( levels_gray2, lines_std2 );

  // show the lines
  const char* lines_pyramid_window_name2 = "LSD lines based on blobs";
  imshow( lines_pyramid_window_name2, levels_gray2 );
  
  printf( "Canny 1 *********************\n" );

  levels.copyTo( src_canny );
  levels_var.copyTo( src_gray_canny );
  /*
  // levels_blobs.copyTo( src_canny );
  dst_canny.create( src_canny.size(), src_canny.type() );
  cvtColor( src_canny, src_gray_canny, COLOR_BGR2GRAY );
  */
  namedWindow( canny_window_name, WINDOW_AUTOSIZE );
  createTrackbar( "Min Threshold:", canny_window_name,
		  &lowThreshold, max_lowThreshold, CannyThreshold );
  CannyThreshold(0, 0);

  printf( "Canny 2 *********************\n" );

  levels.copyTo( src_canny2 );
  dst_canny.create( src_canny2.size(), src_canny2.type() );
  cvtColor( src_canny2, src_gray_canny2, COLOR_BGR2GRAY );
  namedWindow( canny2_window_name, WINDOW_AUTOSIZE );
  createTrackbar( "Min Threshold:", canny2_window_name,
		  &lowThreshold2, max_lowThreshold2, CannyThreshold2 );
  CannyThreshold2(0, 0);
  
  waitKey(0);

  return 0;
}

/***********************************************************************/
