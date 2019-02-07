/************************************************************************/

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

/************************************************************************/

using namespace cv;
using namespace std;

/************************************************************************/

#define MAX_N_COLORS 1000

#define RR 0
#define GG 1
#define BB 2

/************************************************************************/

typedef struct color
{
  double average[3];
  int count;
  uchar replacement[3];
}
  COLOR;

/************************************************************************/

int histogram[256][256][256];

int n_colors = 0;
COLOR color_array[ MAX_N_COLORS + 10 ]; // a little extra for safety

/************************************************************************/

void histogram_colors( Mat& src )
{
  int i, j;
  int r, g, b;
  uchar* p;
  int count = 0;

  // scan the histogram
  for ( r = 0; r < 256; r++ )
    for ( g = 0; g < 256; g++ )
      for ( b = 0; b < 256; b++ )
	histogram[r][g][b] = 0;

  int n_channels = src.channels();

  int nRows = src.rows;
  int nCols = src.cols*n_channels;

  if ( src.isContinuous() )
    {
      nCols *= nRows;
      nRows = 1;
    }

  for( i = 0; i < nRows; ++i )
    {
      p = src.ptr<uchar>(i);
      for ( j = 0; j < nCols; j += n_channels )
	{
	  b = *p++;
	  g = *p++;
	  r = *p++;
	  (histogram[r][g][b])++;
	  count++;
	}
    }
  /*
  printf( "%d %d %d %d: %d %d %d\n",
	  src.rows, src.cols, src.rows*src.cols, src.rows*src.cols*3,
	  nRows, nCols, count );
  */
}

/************************************************************************/

void find_popular_colors( Mat& src )
{
  int h; // histogram index
  int c; // color_index
  int r, g, b;
  double delta_r, delta_g, delta_b, distance2;
  double distance_threshold = 1000.0; // 3000
  bool found_color;
  int count;
  int total = 0;
  
  // scan the histogram
  for ( r = 0; r < 256; r++ )
    {
      for ( g = 0; g < 256; g++ )
	{
	  for ( b = 0; b < 256; b++ )
	    {
	      count = histogram[r][g][b];
	      if ( count == 0 )
		continue;
	      // printf( "%d: %d %d %d\n", count, r, g, b );
	      found_color = false;
	      // scan the list of colors
	      for ( c = 0; c < n_colors; c++ )
		{
		  delta_r = r - color_array[ c ].average[ RR ];
		  delta_g = g - color_array[ c ].average[ GG ];
		  delta_b = b - color_array[ c ].average[ BB ];
		  distance2 =
		    delta_r*delta_r + delta_g*delta_g + delta_b*delta_b;
	      /*
	      printf( "%d: %d %d %d - %d: %g %g %g = %g %g %g, %g\n",
		      count, r, g, b,
		      color_array[ c ].count,
		      color_array[ c ].average[ RR ],
		      color_array[ c ].average[ GG ],
		      color_array[ c ].average[ BB ],
		      delta_r, delta_g, delta_b, distance2 );
	      getchar();
	      */
		  if ( distance2 < distance_threshold )
		    {
		      color_array[ c ].average[ RR ] =
			(color_array[ c ].average[ RR ]*color_array[ c ].count
			 + r*count)/(color_array[ c ].count + count);
		      color_array[ c ].average[ GG ] =
			(color_array[ c ].average[ GG ]*color_array[ c ].count
			 + g*count)/(color_array[ c ].count + count);
		      color_array[ c ].average[ BB ] =
			(color_array[ c ].average[ BB ]*color_array[ c ].count
			 + b*count)/(color_array[ c ].count + count);
		      color_array[ c ].count += count;
		      total += count;
		      found_color = true;
		      break;
		    }
		}
	      if ( found_color )
		continue; // move to next histogram entry
	      // need to add a new color;
	      // printf( "adding %d: %d %d %d\n", count, r, g, b );
	      if ( n_colors < MAX_N_COLORS )
		{
		  color_array[ n_colors ].average[ RR ] = r;
		  color_array[ n_colors ].average[ GG ] = g;
		  color_array[ n_colors ].average[ BB ] = b;
		  color_array[ n_colors ].count = count;
		  total += count;
		  n_colors++;
		}
	    }
	}
    }

  for ( c = 0; c < n_colors; c++ )
    {
      color_array[ c ].replacement[ RR ] = round( color_array[ c ].average[ RR ] );
      color_array[ c ].replacement[ GG ] = round( color_array[ c ].average[ GG ] );
      color_array[ c ].replacement[ BB ] = round( color_array[ c ].average[ BB ] );
    }
  
  printf( "%d colors\n", n_colors );
  /*
  for ( c = 0; c < 20; c++ )
    printf( "%d: %g %g %g\n", color_array[ c ].count,
	    color_array[ c ].average[ RR ],
	    color_array[ c ].average[ GG ],
	    color_array[ c ].average[ BB ] );
  printf( "%d total pixels\n", total );
  */
}

/************************************************************************/
// for now get rid of the most popular color, and rare colors

void edit_colors()
{
  int c;
  int max_count = 0;
  int max_index = -1;

  if ( n_colors == 0 )
    return;

  // scan the list of colors
  for ( c = 0; c < n_colors; c++ )
    {
      color_array[ c ].replacement[ RR ] = round( color_array[ c ].average[ RR ] );
      color_array[ c ].replacement[ GG ] = round( color_array[ c ].average[ GG ] );
      color_array[ c ].replacement[ BB ] = round( color_array[ c ].average[ BB ] );
      if ( color_array[ c ].count > max_count )
	{
	  max_index = c;
	  max_count = color_array[ c ].count;
	}
      if ( color_array[ c ].count < 100 )
	{
	  color_array[ c ].replacement[ RR ] = 0;
	  color_array[ c ].replacement[ GG ] = 0;
	  color_array[ c ].replacement[ BB ] = 0;
	}
    }
  color_array[ max_index ].replacement[ RR ] = 0;
  color_array[ max_index ].replacement[ GG ] = 0;
  color_array[ max_index ].replacement[ BB ] = 0;
}

/************************************************************************/

int find_closest_color( int r, int g, int b, double *distance2_arg )
{
  int c;
  double delta_r, delta_g, delta_b, distance2;
  double min_distance2 = 1e7;
  int min_index = -1;
  
  // scan the list of colors
  for ( c = 0; c < n_colors; c++ )
    {
      delta_r = r - color_array[ c ].average[ RR ];
      delta_g = g - color_array[ c ].average[ GG ];
      delta_b = b - color_array[ c ].average[ BB ];
      distance2 = delta_r*delta_r + delta_g*delta_g + delta_b*delta_b;
      if ( distance2 < min_distance2 )
	{
	  min_distance2 = distance2;
	  min_index = c;
	}
      /*
      printf( "%d: %d %d %d - %g %g %g: %g %g %g: %g; %g %d\n",
	      c, r, g, b, color_array[ c ].average[ RR ],
	      color_array[ c ].average[ GG ],
	      color_array[ c ].average[ BB ],
	      delta_r, delta_g, delta_b, distance2, min_distance2, min_index );
      */
    }
  CV_Assert( min_index >= 0 );
  CV_Assert( min_index < n_colors );

  if ( distance2_arg != NULL )
    *distance2_arg = min_distance2;
  return min_index;
}

/************************************************************************/

void mark_colors( Mat& src, Mat& marked )
{
  int i, j, c;
  int r, g, b;
  uchar *p, *p2;

  int n_channels = src.channels();

  int nRows = src.rows;
  int nCols = src.cols*n_channels;

  if ( src.isContinuous() )
    {
      nCols *= nRows;
      nRows = 1;
    }

  for( i = 0; i < nRows; ++i )
    {
      p = src.ptr<uchar>(i);
      p2 = marked.ptr<uchar>(i);
      for ( j = 0; j < nCols; j += n_channels )
	{
	  b = *p++;
	  g = *p++;
	  r = *p++;
	  c = find_closest_color( r, g, b, NULL );
	  *p2++ = color_array[ c ].replacement[ BB ];
	  *p2++ = color_array[ c ].replacement[ GG ];
	  *p2++ = color_array[ c ].replacement[ RR ];
	}
    }
}

/************************************************************************/

void seed_blobs( Mat& src, Mat& blobs, int *blob_ids  )
{
  int i, j, c;
  int r, g, b;
  uchar *p, *p2;
  double distance2;
  int *p_blob_ids;

  int n_channels = src.channels();

  int nRows = src.rows;
  int nCols = src.cols*n_channels;

  if ( src.isContinuous() )
    {
      nCols *= nRows;
      nRows = 1;
    }

  p_blob_ids = blob_ids;
  for( i = 0; i < nRows; ++i )
    {
      p = src.ptr<uchar>(i);
      p2 = blobs.ptr<uchar>(i);
      for ( j = 0; j < nCols; j += n_channels )
	{
	  b = *p++;
	  g = *p++;
	  r = *p++;
	  c = find_closest_color( r, g, b, &distance2 );
	  if ( distance2 <= 100.0 )
	    {
	      *p2++ = color_array[ c ].replacement[ BB ];
	      *p2++ = color_array[ c ].replacement[ GG ];
	      *p2++ = color_array[ c ].replacement[ RR ];
	      *p_blob_ids++ = c;
	    }
	  else
	    {
	      *p2++ = 0;
	      *p2++ = 0;
	      *p2++ = 0;
	      *p_blob_ids++ = -1;
	    }
	}
    }
}

/************************************************************************/

int check_neighbors( int r, int g, int b, int i, int j, Mat& _src,
		     int *blob_ids )
{

/*
b = _src(i,j)[0];
	  g = _src(i,j)[1];
	  r = _src(i,j)[2];
*/
}

/************************************************************************/

int grow_blobs( Mat& src, Mat& blobs, int *blob_ids  )
{
  int i, j;
  int r, g, b;
  double distance2;
  int n_changes = 0;

  Mat_<Vec3b> _src = src;
  // scan pixels
  for( int i = 0; i < blobs.rows; ++i)
    {
      for( int j = 0; j < blobs.cols; ++j )
	{
	  // skip this pixel if not labelled
	  if ( blob_ids[ i + j*blobs.cols ] == -1 )
	    continue;
	  n_changes += check_neighbors( r, g, b, i, j, _src, blob_ids );
	}
    }
  return n_changes;
}

/************************************************************************/

int main( int argc, char** argv )
{
  Mat src;
  const char* source_window = "Original image";

  CommandLineParser parser( argc, argv,
			    "{@input | planks1ss.jpg | input image}" );
  src = imread( parser.get<String>( "@input" ) );
  if ( src.empty() )
    {
      cout << "Could not open or find the image!\n" << endl;
      cout << "Usage: " << argv[0] << " <Input image>" << endl;
      return -1;
    }
  // accept only char type matrices
  CV_Assert(src.depth() == CV_8U);
  // accept only 3 channel images
  CV_Assert(src.channels() == 3);

  /*
  MatIterator_<Vec3b> it, end;
  for( it = src.begin<Vec3b>(), end = src.end<Vec3b>(); it != end; ++it )
    {
      (*it)[0] = table[(*it)[0]];
      (*it)[1] = table[(*it)[1]];
      (*it)[2] = table[(*it)[2]];
    }
  */

  namedWindow( source_window );
  imshow( source_window, src );

  histogram_colors( src );
  find_popular_colors( src );
  // edit_colors();

  Mat marked = src.clone();
  const char* marked_window = "Marked image";
  mark_colors( src, marked );

  Mat blobs = src.clone();
  const char* blobs_window = "Blobs";
  int blob_ids[ blobs.rows*blobs.cols ];
  seed_blobs( src, blobs, blob_ids );
  grow_blobs( src, blobs, blob_ids );

  imshow( source_window, src );

  namedWindow( marked_window );
  imshow( marked_window, marked );

  namedWindow( blobs_window );
  imshow( blobs_window, blobs );

  waitKey();
  return 0;
}

/************************************************************************/
