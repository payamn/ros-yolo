/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 ThundeRatz

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/console.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <yolo2/ImageDetections.h>

#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <opencv2/core/core.hpp>
#include "cv_bridge/cv_bridge.h"

#include "darknet/yolo2.h"

namespace
{
  darknet::Detector yolo;
  ros::Publisher publisher;
  image_transport::Publisher pub_debug_image_;
  ros::Subscriber sub_enable_;
  image im = {};
  float *image_data = nullptr;
  int imageH, imageW;
  ros::Time timestamp;
  std::mutex mutex;
  std::condition_variable im_condition;
  cv::Mat frame_debug_;
  bool enabled_ = false;

  void imageCallback(const sensor_msgs::ImageConstPtr& msg)
  {
    im = yolo.convert_image(msg);
    std::unique_lock<std::mutex> lock(mutex);
    if (image_data)
      free(image_data);
    timestamp = msg->header.stamp;
    imageH = msg->height;
    imageW = msg->width;
    image_data = im.data;
    im_condition.notify_one();
    try
    {
      frame_debug_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image.clone();
    }
    catch (cv::Exception &e)
    {
      ROS_ERROR_STREAM("E: " << e.what());
    }
    lock.unlock();
  }

  void enableCallback(const std_msgs::BoolConstPtr &enable_msg_ptr)
  {
    if(enabled_ == enable_msg_ptr->data)
      return;
    enabled_ = enable_msg_ptr->data;
    ROS_INFO_STREAM("[YOLO] request: " << (enabled_ ? "enable" : "disable"));
  }
}  // namespace

namespace yolo2
{
  class Yolo2Nodelet : public nodelet::Nodelet
  {
  public:
    virtual void onInit()
    {
      ros::NodeHandle& node = getPrivateNodeHandle();
      const std::string NET_DATA = ros::package::getPath("yolo2") + "/data/";
      std::string config = NET_DATA + "yolo.cfg", weights = NET_DATA + "yolo.weights";
      double confidence, nms;
      node.param<double>("confidence", confidence, .8);
      node.param<double>("nms", nms, .4);
      yolo.load(config, weights, confidence, nms);

      image_transport::ImageTransport transport = image_transport::ImageTransport(node);
      subscriber = transport.subscribe("/camera/rgb/image_rect_color", 1, imageCallback);
      sub_enable_ = node.subscribe("enable", 1, enableCallback);
      pub_debug_image_ = transport.advertise("debug_image", 1);
      publisher = node.advertise<yolo2::ImageDetections>("detections", 5);
      yolo_thread = new std::thread(run_yolo);
    }

    ~Yolo2Nodelet()
    {
      yolo_thread->join();
      delete yolo_thread;
    }

  private:
    image_transport::Subscriber subscriber;
    std::thread *yolo_thread;
    static void run_yolo()
    {
      while (ros::ok())
        {
          if( enabled_)
          {
          float *data;
          ros::Time stamp;
          {
            std::unique_lock<std::mutex> lock(mutex);
            while (!image_data)
              im_condition.wait(lock);
            data = image_data;
            image_data = nullptr;
            stamp = timestamp;
          }
          boost::shared_ptr<yolo2::ImageDetections> detections(new yolo2::ImageDetections);
          *detections = yolo.detect(data, imageH, imageW);
          detections->header.stamp = stamp;
          publisher.publish(detections);
          if (pub_debug_image_.getNumSubscribers() > 0 )
            {
              for( int i = 0; i < detections->detections.size(); i++)
                {
                  cv::rectangle(frame_debug_, cv::Rect(detections->detections[i].roi.x_offset, detections->detections[i].roi.y_offset, detections->detections[i].roi.width, detections->detections[i].roi.height), CV_RGB(0, 128, 128), 2);
                }

              std_msgs::Header h;
              h.stamp = timestamp;
              sensor_msgs::ImagePtr debug_img_msg = cv_bridge::CvImage( h,
                                                                       sensor_msgs::image_encodings::BGR8,
                                                                       frame_debug_).toImageMsg();

              pub_debug_image_.publish(debug_img_msg);
            }
          free(data);
        }
      }
    }
  };
}  // namespace yolo2

PLUGINLIB_EXPORT_CLASS(yolo2::Yolo2Nodelet, nodelet::Nodelet)
