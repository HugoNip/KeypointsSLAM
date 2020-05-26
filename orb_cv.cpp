#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

std::string image_file1 = "../data/1.png";
std::string image_file2 = "../data/2.png";

int main (int argc, char** argv) {
    // Read images
    cv::Mat img_1 = cv::imread(image_file1);
    cv::Mat img_2 = cv::imread(image_file2);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    // Initialization
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // step1: detect Oriented FAST features' location
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // step2: compute BRIEF descriptor
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "extract ORB cost = " << time_used.count() << " seconds." << std::endl;

    cv::Mat outimg1;
    cv::drawKeypoints(img_1, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("ORB features", outimg1);
    cv::imwrite("../results/ORB_features.png", outimg1);

    // step3: matching keypoints, using Hamming distance
    std::vector<cv::DMatch> matches;
    t1 = std::chrono::steady_clock::now();
    matcher->match(descriptors_1, descriptors_2, matches);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "match ORB cost = " << time_used.count() << " seconds." << std::endl;

    // step4: select matching points
    // compute the minimum distance and maximum distance
    auto min_max = std::minmax_element(matches.begin(), matches.end(),
            [](const cv::DMatch &m1, const cv::DMatch &m2) {return m1.distance < m2.distance;});
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("--Max dist: %f \n", max_dist);
    printf("--Min dist: %f \n", min_dist);

    // select good matching points
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; ++i) {
        if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }

    // Draw matching result
    cv::Mat img_match;
    cv::Mat img_goodmatch;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    cv::imshow("all matches", img_match);
    cv::imwrite("../results/all_matches.png", img_match);
    cv::imshow("good matches", img_goodmatch);
    cv::imwrite("../results/good_matches.png", img_goodmatch);
    cv::waitKey(0);

    /*
     * extract ORB cost = 0.339364 seconds.
     * match ORB cost = 0.00114455 seconds.
     * --Max dist: 94.000000
     * --Min dist: 4.000000
     */

    return 0;
}

