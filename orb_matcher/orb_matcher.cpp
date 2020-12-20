#include <limits.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include "orb_matcher/orb_matcher.h"

using std::vector;

namespace orb_matcher{
    
const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(int wid, int hei, float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
    ORBmatcher::mnMaxX = wid;
    ORBmatcher::mnMaxY = hei;

}
bool ORBmatcher::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

void ORBmatcher::AssignFeaturesToGrid(std::vector<cv::KeyPoint> mvKeysUn)
{
    int nReserve = 0.5f*mvKeysUn.size()/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY)){
            mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }
}

vector<size_t> ORBmatcher::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel, std::vector<cv::KeyPoint> kps2) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = std::max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = std::min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = std::max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = std::min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = kps2[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

int ORBmatcher::find_matches(cv::InputArray _k1, cv::InputArray _k2, 
                             cv::Mat mDescriptors1, cv::Mat mDescriptors2, 
                             int windowSize , int type, OutputArray img2_coordinates)
{
    cv::Mat kpx1 = _k1.getMat();
    cv::Mat kpx2 = _k2.getMat();

    std::vector<cv::Point2f> input1;
    for (int x = 0; x < kpx1.rows; x++){
            input1.push_back(cv::Point2f(kpx1.at<cv::Point2f>(x)));
        }

    std::vector<cv::Point2f> input2;
    for (int x = 0; x < kpx2.rows; x++)
            input2.push_back(cv::Point2f(kpx2.at<cv::Point2f>(x)));


    vector<KeyPoint> kps1, kps2;
    cv::KeyPoint::convert(input1, kps1);
    cv::KeyPoint::convert(input2, kps2);
    int nmatches=0;
    vector<int>vnMatches12 = vector<int>(kps1.size(),-1);


    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
    mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);
    // std::cout<<mfGridElementWidthInv<<" "<<mfGridElementHeightInv<<"\n";
    N = kps2.size();
    AssignFeaturesToGrid(kps2);

    vector<int>vMatchedDistance(kps2.size(),INT_MAX);
    vector<int>vnMatches21(kps2.size(),-1);

    
    for(size_t i1=0, iend1=kps1.size(); i1<iend1; i1++)
    {
        cv::KeyPoint kp1 = kps1[i1];
        int level1 = kp1.octave;
        // if(level1>0)
        //     continue;
 
        vector<size_t> vIndices2 = GetFeaturesInArea(kp1.pt.x,kp1.pt.y, windowSize,level1,level1, kps2);
        if(vIndices2.empty())
            continue;
        
        cv::Mat d1 = mDescriptors1.row(i1);
        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = mDescriptors2.row(i2);
            float dist;

            if(type==1)
                dist = DescriptorDistance(d1,d2);
            else if(type==2)
                dist = DescriptorDistanceSIFT(d1,d2);

            
            if(vMatchedDistance[i2]<=dist)
                continue;

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }
        if(bestDist<=INT_MAX)
        {
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                if(vnMatches21[bestIdx2]>=0)
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;

                if(mbCheckOrientation)
                {
                    float rot = kps1[i1].angle-kps2[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }
    // for(int idxx=0;idxx<vnMatches12.size();idxx++)
        // std::cout<<vnMatches12[idxx]<<" ";

    cv::Mat img2x_coordinates = Mat(1,vnMatches12.size(),CV_32SC1,vnMatches12.data());
    img2x_coordinates.copyTo(img2_coordinates);
    // std::cout<<img2x_coordinates;
    return nmatches;
}


void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}



int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

float ORBmatcher::DescriptorDistanceSIFT(const cv::Mat &a, const cv::Mat &b)
{
    double dist_l2  = norm(a,b,NORM_L2);
    // double dist_l2  = norm(a,b,NORM_HAMMING);
    return dist_l2;

}

}
