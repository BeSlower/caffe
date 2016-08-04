#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/data_transformer.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color, 
    const int interpolation, const int resize_mode) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  float short_side_length, rand_length, scale;
  int min_short_length, max_short_length;
  int nh, nw, mode, interp;
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  float interp_ratio;
  caffe_rng_uniform(1, 0.01f, 3.99f, &interp_ratio);
  // if -1,  use random interpolation for images
  interp = (interpolation == -1) ? static_cast<int> (interp_ratio) : interpolation;
  
  mode = resize_mode;
  DLOG(INFO) << "resize mode: " << mode << ", interpolation mode: " << interp;
  // compatible fix for old prototxt
  if ((height*width) == 0 && (height + width) > 0) {
      mode = 1;
  }
  switch (mode)
  {
  case 0:
      if (height > 0 && width > 0) {
          cv::resize(cv_img_origin, cv_img, cv::Size(width, height), 0, 0, interp);
      }      
      else
      {
          cv_img = cv_img_origin;
      }
      break;
  case 1:
      if (height > 0 && width > 0) {
          short_side_length = std::min(height, width);
      }
      else
      {
          short_side_length = (float)(height + width);          
      }
      CHECK_GT(short_side_length, 0) << "The short side length of images must be greater than 0";
      scale = std::min(cv_img_origin.rows, cv_img_origin.cols) / short_side_length;
      nh = (int)(cv_img_origin.rows / scale);
      nw = (int)(cv_img_origin.cols / scale);
      cv::resize(cv_img_origin, cv_img, cv::Size(nw, nh), 0, 0, interp);
      break;
  case 2:
      CHECK_GT(height, 0) << "The height of images must be greater than 0";
      CHECK_GT(width, 0) << "The width of images must be greater than 0";
      float origin_ratio;
      min_short_length = std::min(height, width);
      max_short_length = std::max(height, width);
      // rand_length = min_short_length + Rand(max_short_length - min_short_length + 1);
      caffe_rng_uniform(1, (float) min_short_length, (float) max_short_length, &rand_length);
      origin_ratio = static_cast<float> (cv_img_origin.cols) / static_cast<float> (cv_img_origin.rows);      
      if (origin_ratio < 1) // width is short
      {
          nw = static_cast<int> (rand_length);
          nh = static_cast<int> (nw / origin_ratio);
      }      
      else // height is short
      {
          nh = static_cast<int> (rand_length);
          nw = static_cast<int> (nh * origin_ratio);
      }
      DLOG(INFO) << "The image is resized to the height and width of " << nh << "x" << nw;
      cv::resize(cv_img_origin, cv_img, cv::Size(nw, nh), 0, 0, interp);
      break;
  case 3:
      CHECK_GT(height, 0) << "The height of images must be greater than 0";
      CHECK_GT(width, 0) << "The width of images must be greater than 0";
      float area_ratio, target_ratio, prob_change_aspect, target_area;
      for (int attempt = 0; attempt < 10; ++attempt)
      {
          caffe_rng_uniform(1, 0.15f, 1.0f, &area_ratio);
          caffe_rng_uniform(1, 0.75f, 1.3333f, &target_ratio); // [3/4, 4/3]
          target_area = cv_img_origin.rows * cv_img_origin.cols * area_ratio; //[0.15, 1.0] * area
          nw = static_cast<int> (std::sqrt(target_area) * target_ratio + 0.5f);
          nh = static_cast<int> (std::sqrt(target_area) / target_ratio + 0.5f);
          caffe_rng_uniform(1, 0.0f, 1.0f, &prob_change_aspect);
          if (prob_change_aspect > 0.5)
          {
              int tmp = nw;
              nw = nh;
              nh = tmp;
          }
          if ((nh > 1) && (nw > 1) && (nh <= cv_img_origin.rows) && (nw <= cv_img_origin.cols))
          {
              int h_off = 0;
              int w_off = 0;
              float tmp_off;
              caffe_rng_uniform(1, 0.0f, float(cv_img_origin.rows - nh + 0.99f), &tmp_off);
              h_off = static_cast<int> (tmp_off);
              caffe_rng_uniform(1, 0.0f, float(cv_img_origin.cols - nw + 0.99f), &tmp_off);
              w_off = static_cast<int> (tmp_off);
              cv::Rect roi(w_off, h_off, nw, nh);
              cv::Mat cv_cropped_img = cv_img_origin(roi);
              if ((cv_cropped_img.cols == nw) && (cv_cropped_img.rows == nh)) {
                  DLOG(INFO) << "The image is of size " << cv_img_origin.rows << "x" << cv_img_origin.cols 
                        << ", cropped to " << nh << "x" << nw << " with offset " << h_off << "x" << w_off;
                  cv::resize(cv_cropped_img, cv_img, cv::Size(width, height), 0, 0, interp);
                  return cv_img;
              }
          }
      }
      short_side_length = std::min(height, width);
      CHECK_GT(short_side_length, 0) << "The short side length of images must be greater than 0";
      scale = std::min(cv_img_origin.rows, cv_img_origin.cols) / short_side_length;
      nh = static_cast<int> (cv_img_origin.rows / scale);
      nw = static_cast<int> (cv_img_origin.cols / scale);
      cv::resize(cv_img_origin, cv_img, cv::Size(nw, nh), 0, 0, interp);
      break;
  default:
      LOG(FATAL) << "Unknown resizing mode.";
  }
  
  return cv_img;
}
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color, 
    const int interpolation) {
    return ReadImageToCVMat(filename, height, width, is_color, interpolation, 0);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
    return ReadImageToCVMat(filename, height, width, is_color, 1);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}
// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}
#endif  // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}
#endif  // USE_OPENCV
}  // namespace caffe
