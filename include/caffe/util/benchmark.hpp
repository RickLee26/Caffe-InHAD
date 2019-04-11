#ifndef CAFFE_UTIL_BENCHMARK_H_
#define CAFFE_UTIL_BENCHMARK_H_

#include <boost/date_time/posix_time/posix_time.hpp>

#include "caffe/util/device_alternate.hpp"

namespace caffe {

class Timer {
 public:
  Timer();
  virtual ~Timer();
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();
  virtual float Seconds();

  inline bool initted() { return initted_; }
  inline bool running() { return running_; }
  inline bool has_run_at_least_once() { return has_run_at_least_once_; }

 protected:
  void Init();

  bool initted_;
  bool running_;
  bool has_run_at_least_once_;
#ifndef CPU_ONLY
  cudaEvent_t start_gpu_;
  cudaEvent_t stop_gpu_;
#endif
  boost::posix_time::ptime start_cpu_;
  boost::posix_time::ptime stop_cpu_;
  float elapsed_milliseconds_;
  float elapsed_microseconds_;
};

class CPUTimer : public Timer {
 public:
  explicit CPUTimer();
  virtual ~CPUTimer() {}
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();
};


struct TrainInfo{
  TrainInfo() {
    total_iteration = 0;
    total_info_comm = 0;
    total_diff_comm = 0;
    datum_byte = 0;
    solver_count = 0;
    data_size = 0;
    total_time = 0;
    total_forward_time = 0;
    total_backward_time = 0;
    total_submit_time = 0;
    total_update_time = 0;
  }

  long long infoCommBytes() {
    return total_info_comm * datum_byte * solver_count;
  }

  long long diffCommBytes() {
    return total_diff_comm * datum_byte * data_size;
  }

  int total_iteration;
  int total_info_comm;
  int total_diff_comm;
  int datum_byte;
  int solver_count;
  int data_size;
  float total_time;
  float total_forward_time;
  float total_backward_time;
  float total_submit_time;
  float total_update_time;
};


class InfoCollector : public CPUTimer {
public:
  explicit InfoCollector(string& filetag) : filename(filetag){
    start_times_.clear();
    finish = false;
  }

  virtual void Start() {
    this->start_times_.push_back(boost::posix_time::microsec_clock::local_time());
  }

  virtual void addBatchTime(int step) {
    float tmp_time = getTime();
    switch(step){
      case 0: trainInfo.total_time += tmp_time; break;
      case 1: trainInfo.total_forward_time += tmp_time;break;
      case 2: trainInfo.total_backward_time += tmp_time;break;
      case 3: trainInfo.total_submit_time += tmp_time;break;
      case 4: trainInfo.total_update_time += tmp_time;break;
    }
  }

  virtual void setIteration(int iter) {
    trainInfo.total_iteration = iter;
  }

  virtual void setDataSize(int datum_byte, int solver_count, int data_size) {
    trainInfo.datum_byte = datum_byte;
    trainInfo.solver_count = solver_count;
    trainInfo.data_size = data_size;
  }

  virtual void increInfoComm() {
    trainInfo.total_info_comm++;
  }

  virtual void increDiffComm() {
    trainInfo.total_diff_comm++;
  }

  virtual void calcAndPrint(int iter) {

    LOG(INFO) << "InfoName: " << filename;
    std::fstream op(filename.c_str(), std::fstream::trunc);

    op << "Iteration" << ',' << trainInfo.total_iteration << '\n';
    op << "Iterating" << ',' << iter << '\n';
    op << "InfoComm" << ',' << trainInfo.total_info_comm << '\n';
    op << "DiffComm" << ',' << trainInfo.total_diff_comm << '\n';
    op << "DiffSize" << ',' << trainInfo.data_size << '\n';
    op << "TotalTime" << ',' << trainInfo.total_time << '\n';
    op << "Forward" << ',' << trainInfo.total_forward_time << '\n';
    op << "Backward" << ',' << trainInfo.total_backward_time << '\n';
    op << "Submit" << ',' << trainInfo.total_submit_time << '\n';
    op << "Update" << ',' << trainInfo.total_update_time << '\n';
    op << "Accuracy" << ',' << "undefined" << '\n';

    op.close();

  }

protected:
  virtual float getTime() {
    this->start_cpu_ = start_times_.back();
    start_times_.pop_back();
    this->stop_cpu_ = boost::posix_time::microsec_clock::local_time();
    return (this->stop_cpu_ - this->start_cpu_).total_microseconds();
  }

  // for multi timer
  vector<boost::posix_time::ptime> start_times_;

  TrainInfo trainInfo;

  string filename;

  bool finish;
};


}  // namespace caffe

#endif   // CAFFE_UTIL_BENCHMARK_H_
