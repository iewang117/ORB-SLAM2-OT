//
// Created by qwe on 2020/5/20.
//

#ifndef ORB_SLAM2_OPTICALFLOW_H
#define ORB_SLAM2_OPTICALFLOW_H

#include "Frame.h"
#include "kcftracker.hpp"

#include <opencv2/opencv.hpp>

using namespace cv;

vector<cv::Point2f> lastKeyPoints;
vector<cv::Point2f> trackKeyPoints;
vector<cv::Point2f> currentKeyPoints;
vector<cv::Point2f> movingPoints;
vector<cv::Rect> rects;
vector<cv::Rect> trackRects;
list<KCFTracker> trackers;
vector<float> movingPossibility;
vector<int> movingCounts;
vector<bool> movingFlag;
int normalFrameCounts = 1;
Mat mImRGB;
Mat lastGray;
Mat lastKeyGray;
Mat imgOpticalFlow;
float trackPossib;

Mat TrackWithLK(ORB_SLAM2::KeyFrame *lastKeyFrame, ORB_SLAM2::Frame &currentFrame, vector<cv::Point2f> &mLastKeyPoints, vector<cv::Point2f> &mCurrentKeyPoints, vector<cv::Point2f> &mTrackKeyPoints,
                Mat &mLastGray, Mat &mLastKeyGray, Mat &mCurrentGray, Mat &mCurrentDpeth, vector<cv::Point2f> &mMovingPoints, vector<int> &mMovingCounts, int &mNormalFrameCounts)
{
    movingFlag.clear();
    rects.clear();
    trackRects.clear();

    //第一帧直接返回
    if(mLastGray.empty())
    {
        for(int i=0;i<currentFrame.mvKeys.size();i++)
        {
            movingFlag.push_back(1);
        }
        mLastGray = mCurrentGray.clone();
        return mCurrentGray;
    }

    for(int i=0;i<currentFrame.mvKeys.size();i++)
    {
        mCurrentKeyPoints.push_back(currentFrame.mvKeys[i].pt);
    }

    if(lastKeyFrame->mnFrameId == (currentFrame.mnId-1))
    {
        mLastKeyGray = mLastGray;
        mLastKeyPoints.clear();
        mNormalFrameCounts = 1;

        for(int i=0;i<lastKeyFrame->mvKeys.size();i++)
        {
            mLastKeyPoints.push_back(lastKeyFrame->mvKeys[i].pt);
        }
    } else{
        mNormalFrameCounts++;
    }

    //光流追踪
    vector<uchar> status;
    vector<float> error;
    calcOpticalFlowPyrLK(mLastKeyGray, mCurrentGray, mLastKeyPoints, mTrackKeyPoints, status, error, cv::Size(21,21), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

    //删除丢失点
    int i = 0;
    for (auto it=mLastKeyPoints.begin(); it != mLastKeyPoints.end(); ++i)
    {
        if(!status[i])
        {
            it = mLastKeyPoints.erase(it);
            continue;
        }
        ++it;
    }
    i = 0;
    for (auto it=mTrackKeyPoints.begin(); it != mTrackKeyPoints.end(); ++i)
    {
        if(!status[i])
        {
            it = mTrackKeyPoints.erase(it);
            continue;
        }
        ++it;
    }
    i = 0;
    for (auto it=mMovingCounts.begin(); it!=mMovingCounts.end(); ++i)
    {
        if(!status[i])
        {
            it = mMovingCounts.erase(it);
            continue;
        }
        ++it;
    }

    //K均值聚类
    Scalar colorTab[] = {
            Scalar(0, 0, 255),
            Scalar(0, 255, 0),
            Scalar(255, 0, 0),
            Scalar(0, 255, 255),
            Scalar(255, 0, 255),
            Scalar(255, 255, 0)
    };
    int clusterCount = 10;//分类数
    Mat data;
    Mat center;//聚类后的类别的中心
    Mat labels;//聚类后的标签

    for (int i = 0; i < mTrackKeyPoints.size(); i++) {
        //为聚类向量添加坐标/深度/光流运动矢量等特征
        float depthOfPoint = static_cast<int>(mCurrentDpeth.at<float>(mTrackKeyPoints[i].y, mTrackKeyPoints[i].x) * 10000);
        float deltaX = static_cast<int>((mTrackKeyPoints[i].x - mLastKeyPoints[i].x) * 100);
        float deltaY = static_cast<int>((mTrackKeyPoints[i].y - mLastKeyPoints[i].y) * 100);
        Mat temp = (Mat_<float>(1, 3) << depthOfPoint, deltaX, deltaY);
        data.push_back(temp);
    }

    TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
    kmeans(data, clusterCount, labels, criteria, 1, KMEANS_PP_CENTERS, center);
//    cout << center << endl;

    //排序找到深度偏小的簇
    vector<pair<float,int> > depthSort;
    for(int i = 0; i < clusterCount; i++)
    {
        {
            float depthTemp = center.at<float>(i,0);
            depthSort.push_back(make_pair(depthTemp,i));
        }
    }
    for(int i = 0; i < clusterCount - 1; i++)
    {
        for(int j = 0; j < clusterCount - i - 1; j++)
        {
            if(depthSort[j].first > depthSort[j+1].first)
                swap(depthSort[j],depthSort[j+1]);
        }
    }

    //删除前景主体点
    vector<bool> backgroundFlag;
    vector<Point2f> backKeyPoints;
    vector<Point2f> backLastKeyPoints;
    for (int i = 0; i < mTrackKeyPoints.size(); i++)
    {
        if(labels.at<int>(i) == depthSort[0].second//深度最小的是深度为空的簇
           || labels.at<int>(i) == depthSort[clusterCount-1].second
           || labels.at<int>(i) == depthSort[clusterCount-2].second
           || labels.at<int>(i) == depthSort[clusterCount-3].second
           || labels.at<int>(i) == depthSort[clusterCount-4].second
           || labels.at<int>(i) == depthSort[clusterCount-5].second
           || labels.at<int>(i) == depthSort[clusterCount-6].second
           || labels.at<int>(i) == depthSort[clusterCount-7].second
           || labels.at<int>(i) == depthSort[clusterCount-8].second
           || labels.at<int>(i) == depthSort[clusterCount-9].second)
        {
            backgroundFlag.push_back(1);
            backLastKeyPoints.push_back(mLastKeyPoints[i]);
            backKeyPoints.push_back(mTrackKeyPoints[i]);
        } else{
            backgroundFlag.push_back(0);
        }
    }

    Mat h;
    Mat f;
    Mat maskFund;
    h = findHomography(backLastKeyPoints, backKeyPoints, RANSAC, 3);
    f = findFundamentalMat(backLastKeyPoints, backKeyPoints, CV_FM_RANSAC, 3, 0.99, maskFund);
    int a = 0;
    int b = 0;
    cout << f << endl;
    Mat imgShow = mCurrentGray.clone();
    cvtColor(imgShow, imgShow, CV_GRAY2BGR);
    vector<Vec3f> lines;
    computeCorrespondEpilines(Mat(backLastKeyPoints), 1, f, lines);
    for(int i = 0; i < backKeyPoints.size(); i++)
    {
        Vec3f line = lines[i];
        double dist = fabs(line[0] * backKeyPoints[i].x + line[1] * backKeyPoints[i].y+ line[2])/sqrt(line[0]*line[0]+line[1]*line[1]);
        cout << dist << endl;
        if(dist > 3)
        {
            circle(imgShow, backKeyPoints[i], 2, Scalar(0,255,0), 2);
        }
    }

    cout << a << " " << b << endl;

//    cout << "homography_matrix is " << endl << h << endl;

    Mat homographyImg = Mat::zeros(mCurrentGray.size(), CV_8UC1);

    warpPerspective(mLastKeyGray,homographyImg,h,mLastKeyGray.size());

    vector<cv::Point2f> backforwardPoints;
    calcOpticalFlowPyrLK(mCurrentGray, homographyImg, mCurrentKeyPoints, backforwardPoints, status, error, cv::Size(21,21), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

    //删除反向追踪丢失点
    i = 0;
    for (auto it=backforwardPoints.begin(); it != backforwardPoints.end(); ++i)
    {
        if(!status[i])
        {
            it = backforwardPoints.erase(it);
            continue;
        }
        ++it;
    }
    i = 0;
    for (auto it=mCurrentKeyPoints.begin(); it != mCurrentKeyPoints.end(); ++i)
    {
        if(!status[i])
        {
            it = mCurrentKeyPoints.erase(it);
            continue;
        }
        ++it;
    }

    float distMinTh = 30;
    float distMaxTh = 5000;//阈值
    vector<int> movingPossib;
    for (int i = 0; i < currentKeyPoints.size(); i++)
    {
        float x = backforwardPoints[i].x;
        float y = backforwardPoints[i].y;
        float u = currentKeyPoints[i].x;
        float v = currentKeyPoints[i].y;
        float dist = (u-x)*(u-x)+(v-y)*(v-y);
        if(dist < distMinTh)
        {
            movingFlag.push_back(1);

        }
        else{
            movingFlag.push_back(0);
        }
    }

    //K均值聚类
    int clusterCount2 = 8;//分类数
    Mat data2;
    Mat center2;//聚类后的类别的中心
    Mat labels2;//聚类后的标签

    for (int i = 0; i < mCurrentKeyPoints.size(); i++) {
        //为聚类向量添加坐标/深度/光流运动矢量等特征
        float depthOfPoint = static_cast<int>(mCurrentDpeth.at<float>(mCurrentKeyPoints[i].y, mCurrentKeyPoints[i].x) * 10000);
        float deltaX = static_cast<int>((mCurrentKeyPoints[i].x - backforwardPoints[i].x) * 100);
        float deltaY = static_cast<int>((mCurrentKeyPoints[i].y - backforwardPoints[i].y) * 100);
        float X = static_cast<int>(mCurrentKeyPoints[i].x * 50);
        float Y = static_cast<int>(mCurrentKeyPoints[i].y * 10);
        Mat temp = (Mat_<float>(1, 5) << depthOfPoint, deltaX, deltaY, X, Y);
        data2.push_back(temp);
    }

    kmeans(data2, clusterCount, labels2, criteria, 1, KMEANS_PP_CENTERS, center2);
//    cout << center << endl;

    //排序找到深度偏小的簇
    vector<pair<float,int> > depthSort2;
    for(int i = 0; i < clusterCount; i++)
    {
        {
            float depthTemp = center2.at<float>(i,0);
            depthSort2.push_back(make_pair(depthTemp,i));
        }
    }
    for(int i = 0; i < clusterCount - 1; i++)
    {
        for(int j = 0; j < clusterCount - i - 1; j++)
        {
            if(depthSort2[j].first > depthSort2[j+1].first)
                swap(depthSort2[j],depthSort2[j+1]);
        }
    }

    Mat motionDetect = mCurrentGray.clone();

    cvtColor(motionDetect, motionDetect, CV_GRAY2BGR);

    //绘制稳定点
    for (int i = 0; i < mCurrentKeyPoints.size(); i++)
    {
        if(movingFlag[i])
        {
            Point2f temp;
            int x = mCurrentKeyPoints[i].x;
            int y = mCurrentKeyPoints[i].y;
            temp.x = x;
            temp.y = y;
            circle(motionDetect, temp, 0, Scalar(0,255,0), 1);
        }
//        int index = labels.at<int>(i);
//        circle(imgShow, mCurrentKeyPoints[i], 2, colorTab[index], 2);
    }

    vector<int> numAll(clusterCount,0);
    vector<int> numMoving(clusterCount,0);
    //选择动态率高的cluster
    for (int i = 0; i < mCurrentKeyPoints.size(); i++)
    {
        int y = mCurrentKeyPoints[i].y;
        int x = mCurrentKeyPoints[i].x;
        if(motionDetect.at<cv::Vec3b>(y, x)[0] == 0
        && motionDetect.at<cv::Vec3b>(y, x)[1] == 255
        && motionDetect.at<cv::Vec3b>(y, x)[2] == 0)
        {
            numAll[labels2.at<int>(i)]++;
        }else{
            numAll[labels2.at<int>(i)]++;
            numMoving[labels2.at<int>(i)]++;
        }
    }

    set<int> movingClusters;
    for(int i = 0; i < clusterCount; i++){
        float num1 = numMoving[i];
        float num2 = numAll[i];
        float possib = num1/num2;
        if(possib > 0.8 && num2 > 20 && center2.at<float>(i,0) < 4.5*10000)
        {
            movingClusters.insert(i);
        }
    }

    //绘制结果
    for (int i = 0; i < mCurrentKeyPoints.size(); i++)
    {
        int index = labels2.at<int>(i);
//        if(movingClusters.find(index) != movingClusters.end() && movingClusters.size() < 3)
//            circle(imgShow, mCurrentKeyPoints[i], 2, colorTab[index], 2);
    }

    for (int i = 0; i < clusterCount; i++)
    {
        if(movingClusters.find(i) != movingClusters.end())
        {
            int xmin = INT_MAX;
            int ymin = INT_MAX;
            int xmax = 0;
            int ymax = 0;
            for (int j = 0; j < mCurrentKeyPoints.size(); j++)
            {
                if(labels2.at<int>(j) == i)
                {
                    if(static_cast<int>(mCurrentKeyPoints[j].x) < xmin) xmin = static_cast<int>(mCurrentKeyPoints[j].x);
                    if(static_cast<int>(mCurrentKeyPoints[j].y) < ymin) ymin = static_cast<int>(mCurrentKeyPoints[j].y);
                    if(static_cast<int>(mCurrentKeyPoints[j].x) > xmax) xmax = static_cast<int>(mCurrentKeyPoints[j].x);
                    if(static_cast<int>(mCurrentKeyPoints[j].y) > ymax) ymax = static_cast<int>(mCurrentKeyPoints[j].y);
                }
            }
            rectangle(imgShow, Rect(xmin,ymin,(xmax-xmin),(ymax-ymin)), Scalar(255, 0, 0), 2, 1);
            rects.push_back(Rect(xmin,ymin,(xmax-xmin),(ymax-ymin)));
        }
    }

    cout << "1" << endl;
    while(trackers.size()>5)
    {
        trackers.pop_front();
    }

    cout << "::" << trackers.size() << endl;
    for (auto it=trackers.begin(); it != trackers.end();)
    {
        cout << "2" << endl;
        Rect roi;
        KCFTracker tracker = *it;
        roi = tracker.update(mImRGB, trackPossib);
        if(trackPossib < 0.18)
        {
            cout << "3" << endl;
            trackers.erase(it++);
            cout << "4" << endl;
        }else{
            trackRects.push_back(roi);
            cout << trackPossib << endl;
            rectangle(imgShow, roi, Scalar(255, 0, 0), 2, 1);
            ++it;
        }
//        cout << trackPossib << endl;
//        rectangle(imgShow, roi, Scalar(255, 0, 0), 2, 1);
        cout << "6" << endl;
    }

    for (auto it=rects.begin(); it != rects.end(); ++it)
    {
        bool HOG = true;
        bool FIXEDWINDOW = true;
        bool MULTISCALE = false;
        bool LAB = true;
        KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
        tracker.init(*it, mImRGB);
        trackers.push_back(tracker);
    }

    //刷新特征和图像
    mTrackKeyPoints.clear();
    mCurrentKeyPoints.clear();

    mLastGray = mCurrentGray.clone();

    return imgShow;
}

#endif //ORB_SLAM2_OPTICALFLOW_H