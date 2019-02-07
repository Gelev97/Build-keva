#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
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

  bool useRefine = false;
  bool useCanny = false;
  bool overlay = false;

  Mat image = imread(filename, IMREAD_GRAYSCALE);

  if( image.empty() )
    {
        cout << "Unable to load " << filename;
        return 1;
    }

  imshow("Source Image", image);

  if (useCanny)
    {
        Canny(image, image, 50, 200, 3); // Apply Canny edge detector
    }

  // Create and LSD detector with standard or no refinement.
  Ptr<LineSegmentDetector> ls = useRefine ? createLineSegmentDetector(LSD_REFINE_STD) : createLineSegmentDetector(LSD_REFINE_NONE);

  double start = double(getTickCount());
  vector<Vec4f> lines_std;
  
  // Detect the lines
  ls->detect(image, lines_std);

  double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
  std::cout << "It took " << duration_ms << " ms." << std::endl;

  // Show found lines
  if (!overlay || useCanny)
    {
      image = Scalar(0, 0, 0);
    }

  ls->drawSegments(image, lines_std);

  String window_name = useRefine ? "Result - standard refinement" : "Result - no refinement";
  window_name += useCanny ? " - Canny edge detector used" : "";

  imshow(window_name, image);

  waitKey();
  return 0;
}
