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
        bufs.push((Dtype*)malloc(max_len * sizeof(Dtype)));
      }

      LOG(INFO) << "MessageManager:" << max_len
          << "\t" << mpi_count;
    }

    MPI_Status tryGetOne(Dtype **bufPtr) {

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
      if(bufs.size() == 0) {
        bufs.push((Dtype*)malloc(max_len_ * sizeof(Dtype)));
      }
      return bufs.pop();
    }

    void returnBuffer(Dtype *buf) {
      
      if(buf) {
        bufs.push(buf);
      }
    }

    void markAMessage(MPI_Status stts, Dtype* buf) {
      InfoIn<Dtype> info;
      info.status = stts;
      info.ptr = buf;
      msgs.push(info);
    }

    private:
    int max_len_;
    BlockingQueue<Dtype*> bufs;
    BlockingQueue<InfoIn<Dtype> > msgs;
  };


  template<typename Dtype>
  class GradientTask{
    public:
    GradientTask(Dtype* diff = NULL, int sliceId = 0, int gradId = 0) {
      grad = diff;
      this->sliceId = sliceId;
      this->gradId = gradId;
    }
    Dtype* grad;
    int sliceId;
    int gradId;
  };

  template class BlockingQueue<GradientTask<float> >;
  template class BlockingQueue<GradientTask<double> >;

  template<typename Dtype>
  class GradientManager{

    public:
    GradientManager(int bufCount, int bufSize): 
      bufs(), diffs(), sums(), tasks(){
      this->bufSize = bufSize;
      for(int i = 0; i < bufCount; i++) {
        Dtype* ptr = (Dtype*)malloc(bufSize * sizeof(Dtype));
        bufs.push(ptr);
      }
      dMaxBufCount = 0;
      gDiffCount = 0;
      gDiffResolvCount = 0;
    }

    void setThreadNum(int tnum) {
      threadNum = tnum;
      gGradId = 0;
      gTaskCount = 0;
      sliceSize = bufSize / tnum;
      gradStatus.resize(256, 0);
      cSum = bufs.pop();
      memset(cSum, 0, bufSize * sizeof(Dtype));
    }

    void fetchSum(Dtype* &buf) {
      if(buf) bufs.push(buf);

      omtx.lock();
      buf = sums.pop();
      sums.push(diffs.pop());
      // LOG(INFO) << "DiffResolvCount: " << ++gDiffResolvCount << ":" << diffs.size();
      omtx.unlock();
    }

    GradientTask<Dtype> fetchTask() {
      mtx2.lock();

      if(!tasks.size()) {
        omtx.lock();
        Dtype* diff = diffs.pop();
        gGradId = (gGradId + 1) & 0x00ff;
        for (int i = 0; i < threadNum; i++) {
          GradientTask<Dtype> task(diff, i ,gGradId);
          tasks.push(task);
          gTaskCount++;
        }
        toggleGrad(gGradId, diff, true);
        // gradStatus[gGradId] = threadNum;
        omtx.unlock();
      }

      mtx2.unlock();
      return tasks.pop();
    }

    ~GradientManager(){
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
      GradientTask<Dtype> task;
      while(sums.try_pop(&task));
    }

    // worker submit gradients and get empty buffer
    Dtype* submitNSwap(Dtype* diff) {

      if(diff) {
        diffs.push(diff);
        // LOG(INFO) << "DiffCount: " << ++gDiffCount;
      }

      int diffcount = diffs.size();

      if(diffcount > 2) {
        LOG(INFO) << "bufLen: " << diffcount;
        if(diffcount > dMaxBufCount) {
          dMaxBufCount = diffcount;
          LOG(INFO) << "dMaxBufCount: " << dMaxBufCount;
        }
      }
      
      return bufs.pop();
    }

    void toggleSum(bool show = true) {
      mtx1.lock();

      if(show && !cSum) {
        cSum = sums.pop();
      } 
      
      if(!show) {
        int cnt = --gTaskCount;
        if(cSum && !cnt) {
          sums.push(cSum);
          cSum = NULL;
        }
      }
      mtx1.unlock();
    }

    void toggleGrad(int gid, Dtype* grad, bool reset = false) {
      mtx3.lock();
      if(reset) {
        gradStatus[gid] = threadNum;
      } else if(!--gradStatus[gid]) {
        bufs.push(grad);
        // LOG(INFO) << "DiffResolvCount: " << ++gDiffResolvCount << ":" << diffs.size();
      }
      mtx3.unlock();
    }

    void work(int rank) {
      GradientTask<Dtype> task;

      while(true) {
        task = fetchTask();
        toggleSum();
        
        int offset = sliceSize * task.sliceId;
        int segSize = task.sliceId == threadNum - 1 ? bufSize - offset : sliceSize;
        caffe_mvpy(segSize, task.grad + offset, cSum + offset);

        toggleGrad(task.gradId, task.grad, false);
        toggleSum(false);
        
      }

    }

    private:

    BlockingQueue<Dtype*> bufs, diffs, sums;
    BlockingQueue<GradientTask<Dtype> > tasks;
    int bufSize, dMaxBufCount, threadNum, sliceSize;
    boost::atomic<int> gGradId, gTaskCount, gDiffCount, gDiffResolvCount;
    boost::mutex mtx1, mtx2, mtx3, omtx;
    vector<char> gradStatus;
    Dtype* cSum;
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

    private:
      int mpi_id_, mpi_count_, rate_, max_iter;
      int data_p1_len, total_len, local_solver_count, layer_count;
      shared_ptr<Solver<Dtype> > solver_, test_solver_;
      Dtype *data_, *diff_, *data2;
      shared_ptr<boost::thread> thread_;
      MPI_Datatype type;
      vector<Dtype> iters;
      GradientManager<Dtype> *gb;
      MessageManager<Dtype> *mm;
      BlockingQueue<Dtype*> diffq;
      BlockingQueue<Dtype*> dataq;
      boost::atomic<int> new_comer;
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
  };


  const int MSG_OK = 1;
  const int MSG_ITER = 2;
  const int MSG_HANDIN_DIFF = 4;
  const int MSG_DISPATCH_WEIGHTS = 8;
  const int BIT_DIFF_OVER = 16;

  

#endif




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


}  // namespace caffe

#endif



