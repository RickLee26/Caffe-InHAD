#ifndef CAFFE_PARALLEL_HPP_
#define CAFFE_PARALLEL_HPP_


#include <boost/thread.hpp>
#include <boost/thread/condition.hpp>
#include <boost/atomic.hpp>

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/nccl.hpp"
#include "caffe/util/benchmark.hpp"

#ifdef USE_MPI
#include "mpi.h"
#include <omp.h>
#endif

namespace caffe {

// Represents a net parameters. Once a net is created, its parameter buffers can
// be replaced by ones from Params, to allow parallelization. Params ensures
// parameters are allocated in one consecutive array.
template<typename Dtype>
class Params {
 public:
  explicit Params(shared_ptr<Solver<Dtype> > root_solver);
  virtual ~Params() {
  }

  inline size_t size() const {
    return size_;
  }
  inline Dtype* data() const {
    return data_;
  }
  inline Dtype* diff() const {
    return diff_;
  }

 protected:
  const size_t size_;           // Size of buffers
  Dtype* data_;                 // Network parameters
  Dtype* diff_;                 // Gradient

DISABLE_COPY_AND_ASSIGN(Params);
};


#ifdef USE_NCCL

  // Params stored in GPU memory.
  template<typename Dtype>
  class GPUParams : public Params<Dtype> {
   public:
    GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device);
    virtual ~GPUParams();

    void Configure(Solver<Dtype>* solver) const;

   protected:
    using Params<Dtype>::size_;
    using Params<Dtype>::data_;
    using Params<Dtype>::diff_;
  };

  template<typename Dtype>
  class NCCL : public GPUParams<Dtype>,
               public Solver<Dtype>::Callback,
               public Net<Dtype>::Callback {
   public:
    /**
     * Single process version.
     */
    explicit NCCL(shared_ptr<Solver<Dtype> > solver);
    /**
     * In multi-process settings, first create a NCCL id (new_uid), then
     * pass it to each process to create connected instances.
     */
    NCCL(shared_ptr<Solver<Dtype> > solver, const string& uid);
    ~NCCL();

    boost::barrier* barrier();
    void set_barrier(boost::barrier* value);

    /**
     * In single process settings, create instances without uids and
     * call this to connect them.
     */
    static void InitSingleProcess(vector<NCCL<Dtype>*>* nccls);

    static string new_uid();

    /**
     * Broadcast weights from rank 0 other solvers.
     */
    void Broadcast();

    /**
     * Single process multi-GPU.
     */
    void Run(const vector<int>& gpus, const char* restore);

   protected:
    void Init();
    void on_start() {}
    void run(int layer);  // Net callback
    void on_gradients_ready();

    ncclComm_t comm_;
    cudaStream_t stream_;

    shared_ptr<Solver<Dtype> > solver_;
    // Should not be necessary, https://github.com/NVIDIA/nccl/issues/37
    boost::barrier* barrier_;
    using Params<Dtype>::size_;
    using Params<Dtype>::data_;
    using Params<Dtype>::diff_;
  };

#endif  // USE_NCCL


#ifdef USE_MPI

  template<typename Dtype>
  class InfoIn{
    public:
    MPI_Status status;
    Dtype* ptr;
  };

  template class BlockingQueue<InfoIn<float> >;
  template class BlockingQueue<InfoIn<double> >;

  template<typename Dtype>
  class MessageManager{
    public:
    MessageManager(int max_len, int mpi_count) {
      max_len_ = max_len;
      
      for(int i = 0; i <= mpi_count; i++) {
        // bufs.push_back((Dtype*)malloc(max_len * sizeof(Dtype)));
        bufs.push((Dtype*)malloc(max_len * sizeof(Dtype)));
      }

      LOG(INFO) << "MessageManager:" << max_len
          << "\t" << mpi_count;
    }

    MPI_Status tryGetOne(Dtype **bufPtr) {
      // boost::mutex::scoped_lock lock(mtx);

      // (*bufPtr) = NULL;
      // MPI_Status ret;
      // if(msgs.size() > 0) {
      //   ret = status.front();
      //   status.pop_front();

      //   (*bufPtr) = msgs.front();
      //   msgs.pop_front();
      // }

      InfoIn<Dtype> info;
      info.ptr = NULL;
      msgs.try_pop(&info);
      (*bufPtr) = info.ptr;
      return info.status;
    }

    MPI_Status getOne(Dtype **bufPtr) {
      InfoIn<Dtype> info = msgs.pop();
      (*bufPtr) = info.ptr;
      return info.status;
    }

    Dtype* borrowBuffer() {
      // boost::mutex::scoped_lock lock(mtx);
      // Dtype *buf = NULL;
      // if (bufs.size() == 0) {
      //   bufs.push_back((Dtype*)malloc(max_len_ * sizeof(Dtype)));
      // }
      
      // buf = bufs.front();
      // bufs.pop_front();

      // if(!buf) {
      //   LOG(INFO) << "borrowBuffer: " << bufs.size();
      // }
      if(bufs.size() == 0) {
        bufs.push((Dtype*)malloc(max_len_ * sizeof(Dtype)));
      }
      return bufs.pop();
    }

    void returnBuffer(Dtype *buf) {
      // boost::mutex::scoped_lock lock(mtx);
      // bufs.push_back(buf);
      if(buf) {
        bufs.push(buf);
      }
    }

    void markAMessage(MPI_Status stts, Dtype* buf) {
      // boost::mutex::scoped_lock lock(mtx);
      // status.push_back(stts);
      // msgs.push_back(buf);
      InfoIn<Dtype> info;
      info.status = stts;
      info.ptr = buf;
      msgs.push(info);
    }

    private:
    int max_len_;
    // boost::mutex mtx;
    BlockingQueue<Dtype*> bufs;
    BlockingQueue<InfoIn<Dtype> > msgs;
    // list<Dtype*> bufs;
    // list<MPI_Status> status;
    // list<Dtype*> msgs;
  };


  template<typename Dtype>
  class GradientBuffer{

    public:
    GradientBuffer(int bufCount, int bufSize): bufs(), diffs(), sums(){
      this->bufSize = bufSize;
      for(int i = 0; i < bufCount; i++) {
        Dtype* ptr = genBuf(bufSize, false);
        bufs.push(ptr);
      }
      maxBufCount = 0;
      sumrequire = 0;
      workerindex = 0;
      sum[0] = genBuf(bufSize);
      sum[1] = genBuf(bufSize);
      // sum[1] = NULL;
      workercount[0] = workercount[1] = 0;
    }

    Dtype* genBuf(int bufSize, bool reset = true) {
      Dtype* buf = (Dtype*)malloc(bufSize * sizeof(Dtype));
      if (reset) memset(buf, 0, sizeof(Dtype) * bufSize);
      return buf;
    }

    ~GradientBuffer(){
      Dtype* ptr = NULL;
      while(bufs.try_pop(&ptr)) {
        delete[] ptr;
      }
      while(diffs.try_pop(&ptr)) {
        delete[] ptr;
      }
      while(sums.try_pop(&ptr)) {
        delete[] ptr;
      }
    }

    // worker submit gradients and get empty buffer
    Dtype* submitNSwap(Dtype* diff) {

      if(diff) diffs.push(diff);

      int diffcount = diffs.size();

      if(diffcount > 2) {
        LOG(INFO) << "bufLen: " << diffcount;
        if(diffcount > maxBufCount) {
          maxBufCount = diffcount;
          LOG(INFO) << "maxBufCount: " << maxBufCount;
        }
      }
      
      return bufs.pop();
    }

    Dtype* dealNReturn(Dtype* buf) {
      if(buf) bufs.push(buf);
      Dtype *ret = diffs.pop();
      return ret;
    }

    Dtype* swapCurrent(Dtype* buf) {
      if(!buf) buf = genBuf(bufSize, false);

      workerindex = (workerindex + 1) & 1;

      sumrequire = 1;
      Dtype* sumbuf = sums.pop();
      sumrequire = 0;

      caffe_copy(bufSize, sumbuf, buf);
      memset(sumbuf, 0, bufSize * sizeof(Dtype));
      return buf;
    }

    void work() {
      
      Dtype* curbuf = NULL;

      while(true){

        curbuf = dealNReturn(curbuf);
        
        int idx = workerindex;
        workercount[idx]++;
        Dtype* sumbuf = sum[idx];

        caffe_mvpy(bufSize, curbuf, sumbuf);

        if ((--workercount[idx]) == 0 && sumrequire && sums.size() == 0) {
          sums.push(sumbuf);
        }
        
      }
    }

    
    Dtype* sum[2];
    boost::atomic<int> sumrequire, workercount[2], workerindex;

    private:
    BlockingQueue<Dtype*> bufs, diffs, sums;
    int bufSize, maxBufCount;
  };

  template<typename Dtype>
  class PSManager {
    public:
      PSManager(shared_ptr<Solver<Dtype> > root, int rate);

      // for server
      void listen();
      void run_test(int *flag, int *iters);

      // for client
      void run();
      void* send_recv(void* ptr, int tag, int msglen, MPI_Status& status);

      Dtype* submit_diff(int rank, Dtype* diff);

      shared_ptr<InfoCollector> collector;

    private:
      int mpi_id_, mpi_count_, rate_, max_iter;
      int data_p1_len, total_len, local_solver_count, layer_count;
      shared_ptr<Solver<Dtype> > solver_, test_solver_;
      Dtype *data_, *diff_, *data2;
      shared_ptr<boost::thread> thread_;
      MPI_Datatype type;
      vector<Dtype> iters;
      GradientBuffer<Dtype> *gb;
      MessageManager<Dtype> *mm;
      BlockingQueue<Dtype*> diffq;
      BlockingQueue<Dtype*> dataq;
      boost::atomic<int> new_comer, finish_count;
      boost::condition cond_iter;
      boost::mutex mtx_iter;
      bool debug;
  };

  template<typename Dtype>
  class PSWorker :  public Params<Dtype>,
                    public Solver<Dtype>::Callback {
    public:
      PSWorker(shared_ptr<Solver<Dtype> > root_solver,
                Dtype** g_data,
                Dtype** g_diff,
                int omp_rank,
                int *global_version,
                PSManager<Dtype>* manager
                // Dtype** g_diff,
                // vector<int>* iters,
                // int *new_comer,
                // boost::atomic<int> **pcopier,
                // boost::condition *wt1,
                // boost::condition *wt2
                );

      void work();

    protected:
      // for solver
      void on_start();
      void on_gradients_ready();


    private:
      using Params<Dtype>::size_;
      using Params<Dtype>::data_;
      using Params<Dtype>::diff_;
      Dtype **g_data_;
      Dtype *data_ptr_;

      shared_ptr<Solver<Dtype> > solver_;
      PSManager<Dtype>* manager_;
      int omp_rank_;
      int *global_version_;
      int local_version_;

      shared_ptr<InfoCollector> collector;
  };


  const int MSG_OK = 1;
  const int MSG_ITER = 2;
  const int MSG_HANDIN_DIFF = 4;
  const int MSG_DISPATCH_WEIGHTS = 8;
  const int BIT_DIFF_OVER = 16;

  

#endif

}  // namespace caffe

#endif