#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "orb_extractor/orb_extractor.h"
#include "orb_matcher/orb_matcher.h"


using namespace cv;
using namespace std;
using namespace orb;
using namespace orb_matcher;

std::string img1str = "../EUROC1.png";
std::string img2str = "../EUROC2.png";
int main()
{
    cv::Mat mDescriptors1, mDescriptors2, imout1,imout2, match_out ;
    cv::Mat mvKeys1, mvKeys2;
    ORBextractor mpIniORBextractor = ORBextractor(1024,1.2,8,20,5);

    cv::Mat im1 = cv::imread(img1str);
    cv::Mat im2 = cv::imread(img2str);
    
    if(im1.empty() || im2.empty())
        cout<<"read image error! \n"<<img1str<<"\n"<<img2str<<endl;
    cv::cvtColor(im1, im1,  cv::COLOR_BGR2GRAY);
    cv::cvtColor(im2, im2,  cv::COLOR_BGR2GRAY);
    
    auto t0 = chrono::steady_clock::now();
    mpIniORBextractor.extract_orb_fts(im1,cv::Mat(),mvKeys1,mDescriptors1);
    auto t1 = chrono::steady_clock::now();
    mpIniORBextractor.extract_orb_fts(im2,cv::Mat(),mvKeys2,mDescriptors2);

    vector<int> matches12;

    ORBmatcher ob = ORBmatcher(im1.cols,im1.rows,0.6,true);
    auto t2 = chrono::steady_clock::now();

    int mtchs = ob.find_matches(mvKeys1, mvKeys2,mDescriptors1, mDescriptors2, 100, 1, matches12);
    
    vector<cv::DMatch> dmatches;
    for(int i = 0; i<matches12.size();i++)
    {
        if(matches12[i]!=-1)
        {
            cv::DMatch dm(i, matches12[i], 1);
            dmatches.push_back(dm);
        }
    }

    std::vector<cv::Point2f> input1;
    for (int x = 0; x < mvKeys1.rows; x++)
        input1.push_back(cv::Point2f(mvKeys1.at<cv::Point2f>(x)));
        
    std::vector<cv::Point2f> input2;
    for (int x = 0; x < mvKeys2.rows; x++)
        input2.push_back(cv::Point2f(mvKeys2.at<cv::Point2f>(x)));

    vector<KeyPoint> kps1, kps2;
    cv::KeyPoint::convert(input1, kps1);
    cv::KeyPoint::convert(input2, kps2);
    std::vector<char> mask(mtchs, 1);
    
    cv::drawMatches(im1,kps1, im2, kps2, dmatches, match_out, 
                    cv::Scalar::all(-1), cv::Scalar::all(-1), mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    auto t3 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t3 - t0);
    cout<<"time used: "<<time_used.count()<<endl;
    cv::drawKeypoints(im1, kps1, imout1);
    cv::drawKeypoints(im2, kps2, imout2);
    cv::imshow("Image1", imout1);
    cv::imshow("Image2", imout2);
    cv::imshow("Out", match_out);
    cv::waitKey();

    cv::destroyAllWindows();

    return -1;
}