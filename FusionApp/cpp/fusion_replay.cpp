#include <opencv2/opencv.hpp>

int main() {
  // Load an image from file
  cv::Mat image = cv::imread("img.png");

  // Check if the image was loaded successfully
  if (image.empty()) {
    std::cerr << "Error: Could not load image." << std::endl;
    return -1;
  }

  // Display the image in a window
  cv::imshow("Display Window", image);

  // Wait for a key press indefinitely
  cv::waitKey(0);

  return 0;
}