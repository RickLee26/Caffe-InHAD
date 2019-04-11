#include <glog/logging.h>
#include <stdio.h>
#include <sstream>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"
#include "caffe/sgd_solvers.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/blocking_queue.hpp"

#ifdef USE_NCCL
#include <cuda_runtime.h>
#endif

#ifdef USE_MPI
#include "mpi.h"
#endif

int min(int a, int b){
  return a < b ? a : b;
}

int max(int a, int b){
  return a > b ? a : b;
}

namespace caffe {

enum Op {
  copy,
  replace_cpu,
  replace_gpu,
  replace_cpu_diff,
  replace_gpu_diff
};

template<typename Dtype>
static void apply_buffers(const vector<Blob<Dtype>*>& blobs,
                          Dtype* buffer, size_t total_size, Op op) {
  Dtype* ptr = buffer;
  for (int i = 0; i < blobs.size(); ++i) {
    int size = blobs[i]->count();
    switch (op) {
      case copy: {
        // Init buffer to current values of blobs
        caffe_copy(size,
                   reinterpret_cast<const Dtype*>(blobs[i]->data()->cpu_data()),
                   ptr);
        break;
      }
      case replace_cpu:
        blobs[i]->data()->set_cpu_data(ptr);
        break;
      case replace_gpu:
        blobs[i]->data()->set_gpu_data(ptr);
        break;
      case replace_cpu_diff:
        blobs[i]->diff()->set_cpu_data(ptr);
        break;
      case replace_gpu_diff:
        blobs[i]->diff()->set_gpu_data(ptr);
        break;
    }
    ptr += size;
  }
  // total_size is at least one byte
  CHECK_EQ(total_size, (ptr == buffer ? 1 : ptr - buffer));
}

// Buffer size necessary to store given blobs
template<typename Dtype>
static size_t total_size(const vector<Blob<Dtype>*>& params) {
  size_t size = 0;
  for (int i = 0; i < params.size(); ++i)
    size += params[i]->count();
  // Size have at least one byte, otherwise cudaMalloc fails if net has no
  // learnable parameters.
  return (size > 0) ? size : 1;
}

template<typename Dtype>
Params<Dtype>::Params(shared_ptr<Solver<Dtype> > root_solver)
  : size_(total_size<Dtype>(root_solver->net()->learnable_params())),
    data_(),
    diff_() {
}

INSTANTIATE_CLASS(Params);

#ifdef USE_MPI
  template<typename Dtype>
  PSWorker<Dtype>::PSWorker(shared_ptr<Solver<Dtype> > solver,
                Dtype** g_data,
                Dtype** g_diff,
                int omp_rank,
                int *global_version,
                PSManager<Dtype>* manager
                ) : 
                Params<Dtype>(solver),
                g_data_(g_data),
                solver_(solver),
                omp_rank_(omp_rank),
                global_version_(global_version),
                local_version_(0),
                manager_(manager)
                {
                  data_ = (Dtype*)malloc(size_ * sizeof(Dtype));
                  diff_ = (Dtype*)malloc((size_ + 1) * sizeof(Dtype));

                  if(omp_rank_) {
                    // create worker solver
                    SolverParameter param(solver_->param());
                    param.set_type(solver_->type());
                    int iter = solver_->iter_;
                    solver_.reset(SolverRegistry<Dtype>::CreateSolver(param));
                    solver_->iter_ = iter;

                  }

                  // replace buffer
                  const vector<Blob<Dtype>*>& net = solver_->net()->learnable_params();
                  caffe_copy(size_, *g_data_, data_);
                  apply_buffers(net, data_, size_, replace_cpu);
                  apply_buffers(net, diff_ + 1, size_, replace_cpu_diff);
                  solver_->add_callback(this);
  }

  
  template<typename Dtype>
  void PSWorker<Dtype>::work() {
    solver_->Solve();
  }

  template<typename Dtype>
  void PSWorker<Dtype>::on_start(){
    if(local_version_ < (*global_version_)){
      caffe_copy(size_, *g_data_, data_);
      local_version_ = *global_version_;
    }
  }

  template<typename Dtype>
  void PSWorker<Dtype>::on_gradients_ready(){
    diff_[0] = 1;
    diff_ = manager_->submit_diff(omp_rank_, diff_);
    const vector<Blob<Dtype>*>& net = solver_->net()->learnable_params();
    apply_buffers(net, diff_ + 1, size_, replace_cpu_diff);
  }


  template<typename Dtype>
  PSManager<Dtype>::PSManager(shared_ptr<Solver<Dtype> > root, int rate) {
    mpi_id_ = Caffe::header().mpi_id;
    mpi_count_ = Caffe::header().mpi_count;
    solver_ = root;
    rate_ = rate;

    local_solver_count = Caffe::solver_count() / (mpi_count_ - 1);
    data_p1_len = total_size(root->net()->learnable_params()) + 1;


    total_len = max(local_solver_count, data_p1_len);

    max_iter = solver_->param().max_iter();

    data_ = (Dtype*)malloc(total_len * sizeof(Dtype));
    diff_ = (Dtype*)malloc(total_len * sizeof(Dtype));
    data2 = (Dtype*)malloc(total_len * sizeof(Dtype));

    apply_buffers(root->net()->learnable_params(), data_, data_p1_len - 1, copy);
    apply_buffers(root->net()->learnable_params(), data_, data_p1_len - 1, replace_cpu);
    apply_buffers(root->net()->learnable_params(), diff_ + 1, data_p1_len - 1, replace_cpu_diff);
    caffe_set(total_len, Dtype(0), diff_);

    type = sizeof(Dtype) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;

    debug = false;

    // Sync weights
    LOG(INFO) << "sync weights: " << mpi_id_ << " : " << mpi_count_;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(data_, data_p1_len - 1, type, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if(mpi_id_)run();
    else listen();
    delete [] data_;
    delete [] diff_;
    delete [] data2;
  }


  template<typename Dtype>
  void PSManager<Dtype>::listen() {

    int start_iter = solver_->iter();
    iters.resize(Caffe::solver_count(), start_iter);
    vector<int> updates(mpi_count_ - 1, start_iter);
    vector<int> gathers(mpi_count_ - 1, start_iter);


    int gather_stamp = start_iter, update_stamp = start_iter;
    int gather_count = 0;
    int global_version = 0;
    bool gather_sig = false;

    int test_flag = 1;

    thread_.reset(new boost::thread(&PSManager<Dtype>::run_test, this, &test_flag, &update_stamp));


    int omp_count = min(20, mpi_count_);
    mm = new MessageManager<Dtype>(total_len, mpi_count_);
    boost::condition cond, bufferCond;
    boost::mutex mtx, bufferMtx, sendMtx;
    int finish = 0;

    Dtype *sendBuff = (Dtype*)malloc(total_len * sizeof(Dtype));
    caffe_copy(data_p1_len, data_, sendBuff);
    dataq.push(sendBuff);
    diffq.push(diff_);

    #pragma omp parallel num_threads(omp_count)
    {
      MPI_Status status;
      int tag = 0, dest = 0, msglen = 0;
      int omp_rank = omp_get_thread_num();
      Dtype* buf = NULL;
      while(true) {
        if(!omp_rank) {
          buf = mm->borrowBuffer();
          if(!buf) {
            LOG(ERROR) << "mother fucker!!!!!";
          }
          MPI_Recv(buf, total_len, type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

          if(status.MPI_TAG & MSG_HANDIN_DIFF && ++gather_count == mpi_count_ - 1) {
            status.MPI_TAG |= BIT_DIFF_OVER;
            gather_count = 0;
            gather_sig = false;
            if(gather_stamp == max_iter) {
              finish |= 1;
            }
          }

          mm->markAMessage(status, buf);
          
          if(finish & 1) {
            LOG(INFO) << "finish 1";
            break;
          }
          continue;
        }


        status = mm->getOne(&buf);

        dest = status.MPI_SOURCE;
        tag = MSG_OK;
        msglen = 1;

        if(status.MPI_TAG & MSG_ITER){

          int start = (dest - 1) * local_solver_count;

          for(int i = 0; i < local_solver_count; i++) {
            iters[start + i] = buf[i];
          }

          if(!gather_sig){
            int cnt = 0, acnt = 0;
            for(int i = 0; i < iters.size(); i++) {
              if(iters[i] > gather_stamp)cnt++;
              if(iters[i] == max_iter) acnt++;
            }

            if(cnt * 100 >= rate_ * iters.size()) {
              gather_sig = true;
              gather_stamp++;
            }
            if(acnt == iters.size()) {
              gather_stamp = max_iter;
            }
          }
        } else if(status.MPI_TAG & MSG_HANDIN_DIFF){

          gathers[dest - 1] = gather_stamp;

          Dtype* tmpdiff = diffq.pop();
          
          if(buf[0] > 0.9) caffe_add(data_p1_len, tmpdiff, buf, tmpdiff);

          if(status.MPI_TAG & BIT_DIFF_OVER && tmpdiff[0] > 0.9){

            update_stamp++;

            Dtype bufcnt = tmpdiff[0];
            
            caffe_scal(data_p1_len - 1, (Dtype) 1.0 / tmpdiff[0], tmpdiff + 1);

            solver_->iter_ = gather_stamp;
            LOG(INFO) << "update: " << gather_stamp << "\t" << bufcnt;
            solver_->ApplyUpdate();
            solver_->net()->ClearParamDiffs();
            global_version++;
            tmpdiff[0] = 0;

            Dtype* sendBuff = dataq.pop();
            caffe_copy(data_p1_len - 1, data_, sendBuff);
            dataq.push(sendBuff);
          }
          diffq.push(tmpdiff);
        }


        if(gather_sig && gathers[dest - 1] < gather_stamp) tag |= MSG_HANDIN_DIFF;
        if(updates[dest - 1] < update_stamp) {
          tag |= MSG_DISPATCH_WEIGHTS;
          msglen = data_p1_len - 1;
          updates[dest - 1] = update_stamp;
        }

        msglen = total_len;
        Dtype* sendBuff = dataq.pop();
        MPI_Send(sendBuff, msglen, type, dest, tag, MPI_COMM_WORLD);
        dataq.push(sendBuff);
        mm->returnBuffer(buf);
        if(!gather_sig && gather_stamp == max_iter){
          finish |= 2;
          LOG(INFO) << "finish " << gather_stamp << "\t " << max_iter;
          solver_->iter_ = max_iter;
          break;
        }

      }

      LOG(INFO) << "onFinish";
    }

    LOG(INFO) << "after deal";
    test_flag = 0;
    thread_->join();
    solver_->net()->ClearParamDiffs();
    Caffe::set_solver_rank(0);
    Caffe::set_solver_count(1);
    solver_->TestAll();
    solver_->Snapshot();
  }


  template<typename Dtype>
  void PSManager<Dtype>::run_test(int *flag, int *iters) {
    Caffe::set_solver_count(1);
    Caffe::set_solver_rank(0);
    Caffe::header().mpi_id = 0;
    Caffe::set_mode(Caffe::CPU);

    SolverParameter param(solver_->param());
    param.set_type(solver_->type());
    test_solver_.reset(SolverRegistry<Dtype>::CreateSolver(param));
    apply_buffers(test_solver_->net()->learnable_params(), data2, data_p1_len - 1, replace_cpu);
    // apply_buffers(test_solver_->net()->learnable_params(), diff2 + 1, data_p1_len - 1, replace_cpu_diff);

    while(*flag){
      test_solver_->iter_ = solver_->iter();
      Dtype* sendBuff = dataq.pop();
      caffe_copy(data_p1_len - 1, sendBuff, data2);
      dataq.push(sendBuff);
      test_solver_->net()->ClearParamDiffs();
      test_solver_->TestAll();
    }
  }

  template<typename Dtype>
  void PSManager<Dtype>::run() {

    Dtype *wdata = data_;
    Dtype *diff = diff_;

    iters.resize(local_solver_count, solver_->iter_);

    new_comer = 0;

    // boost::lock_guard<boost::mutex> lk0(mtx0), lk1(mtx1);

    vector<shared_ptr<PSWorker<Dtype> > > vworker(local_solver_count);

    int start = (mpi_id_ - 1) * local_solver_count;
    Caffe::WorkerHeader wh = Caffe::header();
    int solver_count = Caffe::solver_count();
    Dtype* buf = (Dtype*)malloc(total_len * sizeof(Dtype));
    gb = new GradientManager<Dtype>(5, data_p1_len);
    int global_version = 0;

    int totalThreadCount = local_solver_count + 1 + 8;
    gb->setThreadNum(totalThreadCount - local_solver_count - 1);

    #pragma omp parallel num_threads(totalThreadCount)
    {
      int rank = omp_get_thread_num();
      if(rank < local_solver_count){
        Caffe::header() = wh;
        Caffe::header().worker_index = rank;
        Caffe::set_solver_rank(start + rank);
        Caffe::set_solver_count(solver_count);
        vworker[rank].reset(new PSWorker<Dtype>(solver_, &wdata, &diff, rank, &global_version, this));
        vworker[rank]->work();
      } else if (rank == local_solver_count) {
        int tag, msglen;
        
        MPI_Status status;

        bool all_finish = false;

        while(!all_finish){

          if(!new_comer)cond_iter.wait(mtx_iter);

          tag = MSG_ITER;
          msglen = local_solver_count;

          all_finish = true;
          for(int i = 0; i < local_solver_count; i++){
            buf[i] = iters[i];
            if(iters[i] < max_iter)all_finish = false;
          }

          void* ptr = send_recv(buf, tag, msglen, status);

          if(status.MPI_TAG & MSG_DISPATCH_WEIGHTS){
            Dtype *dbuf = wdata == data_ ? data2 : data_;
            memcpy(dbuf, ptr, (data_p1_len - 1) * sizeof(Dtype));
            wdata = dbuf;
            global_version++;
            status.MPI_TAG ^= MSG_DISPATCH_WEIGHTS;
          }

          new_comer = 0;

          if(all_finish || (status.MPI_TAG & MSG_HANDIN_DIFF)){
            // LOG(INFO) << "ohehe";
            gb->fetchSum(diff);
            ptr = send_recv(diff, MSG_HANDIN_DIFF, data_p1_len, status);
          }

          if(status.MPI_TAG & MSG_DISPATCH_WEIGHTS){
            Dtype *dbuf = wdata == data_ ? data2 : data_;
            memcpy(dbuf, ptr, (data_p1_len - 1) * sizeof(Dtype));
            wdata = dbuf;
            global_version++;
          }
        }
      } else {
        gb->work(rank - local_solver_count - 1);
      }
    }

  }

  template<typename Dtype>
  Dtype* PSManager<Dtype>::submit_diff(int rank, Dtype* diff){
    iters[rank]++;
    new_comer++;
    cond_iter.notify_all();
    return gb->submitNSwap(diff);
  }

  template<typename Dtype>
  void* PSManager<Dtype>::send_recv(void* ptr, int tag, int msglen, MPI_Status& status){
    // tag for different type of info: 0x00 for iter

    // msglen = total_len;
    MPI_Send(ptr, msglen, type, 0, tag, MPI_COMM_WORLD);
    MPI_Recv(ptr, total_len, type, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    return ptr;
  }


  INSTANTIATE_CLASS(PSWorker);
  INSTANTIATE_CLASS(PSManager);

#endif // USE_MPI


#ifdef USE_NCCL
  template<typename Dtype>
  GPUParams<Dtype>::GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device)
    : Params<Dtype>(root_solver) {
    int initial_device;
    CUDA_CHECK(cudaGetDevice(&initial_device));

    // Allocate device buffers
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(Dtype)));

    // Copy blob values
    const vector<Blob<Dtype>*>& net =
      root_solver->net()->learnable_params();
    apply_buffers(net, data_, size_, copy);

    CUDA_CHECK(cudaMalloc(&diff_, size_ * sizeof(Dtype)));
    caffe_gpu_set(size_, Dtype(0), diff_);

    CUDA_CHECK(cudaSetDevice(initial_device));
  }

  template<typename Dtype>
  GPUParams<Dtype>::~GPUParams() {
    CUDA_CHECK(cudaFree(data_));
    CUDA_CHECK(cudaFree(diff_));
  }

  template<typename Dtype>
  void GPUParams<Dtype>::Configure(Solver<Dtype>* solver) const {
    const vector<Blob<Dtype>*>& net =
      solver->net()->learnable_params();
    apply_buffers(net, data_, size_, replace_gpu);
    apply_buffers(net, diff_, size_, replace_gpu_diff);
  }

  static int getDevice() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
  }

  template<typename Dtype>
  NCCL<Dtype>::NCCL(shared_ptr<Solver<Dtype> > solver)
    : GPUParams<Dtype>(solver, getDevice()),
      comm_(), solver_(solver), barrier_() {
    this->Configure(solver.get());
    Init();
  }

  template<typename Dtype>
  NCCL<Dtype>::NCCL(shared_ptr<Solver<Dtype> > solver, const string& uid)
    : GPUParams<Dtype>(solver, getDevice()),
      solver_(solver), barrier_() {
    this->Configure(solver.get());
    Caffe::set_multiprocess(true);
    ncclUniqueId nccl_uid;
    memcpy(&nccl_uid, &uid[0], NCCL_UNIQUE_ID_BYTES);  // NOLINT(caffe/alt_fn)
    NCCL_CHECK(ncclCommInitRank(&comm_,
                                Caffe::solver_count(),
                                nccl_uid,
                                Caffe::solver_rank()));
    Init();
  }

  template<typename Dtype>
  void NCCL<Dtype>::Init() {
    if (solver_->param().layer_wise_reduce()) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }
  }

  template<typename Dtype>
  NCCL<Dtype>::~NCCL() {
    if (solver_->param().layer_wise_reduce()) {
      CUDA_CHECK(cudaStreamDestroy(stream_));
    }
    if (comm_) {
      ncclCommDestroy(comm_);
    }
  }

  template<typename Dtype>
  boost::barrier* NCCL<Dtype>::barrier() {
    return barrier_;
  }
  template<typename Dtype>
  void NCCL<Dtype>::set_barrier(boost::barrier* value) {
    barrier_ = value;
  }

  template<typename Dtype>
  void NCCL<Dtype>::InitSingleProcess(vector<NCCL<Dtype>*>* nccls) {
    ncclComm_t* comms = new ncclComm_t[nccls->size()];
    int* gpu_list = new int[nccls->size()];
    for (int i = 0; i < nccls->size(); ++i) {
      gpu_list[i] = (*nccls)[i]->solver_->param().device_id();
    }
    NCCL_CHECK(ncclCommInitAll(comms, static_cast<int>(nccls->size()), gpu_list));
    for (int i = 0; i < nccls->size(); ++i) {
      (*nccls)[i]->comm_ = comms[i];
    }
  }

  template<typename Dtype>
  string NCCL<Dtype>::new_uid() {
    string uid;
    uid.resize(NCCL_UNIQUE_ID_BYTES);
    ncclUniqueId nccl_uid;
    NCCL_CHECK(ncclGetUniqueId(&nccl_uid));
    memcpy(&uid[0], &nccl_uid, NCCL_UNIQUE_ID_BYTES);  // NOLINT(caffe/alt_fn)
    return uid;
  }

  template<typename Dtype>
  void NCCL<Dtype>::Broadcast() {
    if (barrier_) {  // NULL in multi process case
      barrier_->wait();
    }
    NCCL_CHECK(ncclBcast(data_, static_cast<int>(size_),
                         nccl::dataType<Dtype>::type, 0,
                         comm_, cudaStreamDefault));
    if (barrier_) {
      barrier_->wait();
    }
  }

  template<typename Dtype>
  void NCCL<Dtype>::run(int layer) {
    CHECK(solver_->param().layer_wise_reduce());
    vector<shared_ptr<Blob<Dtype> > >& blobs =
      solver_->net()->layers()[layer]->blobs();
  #ifdef DEBUG
    // Assert blobs are contiguous to reduce in one step (e.g. bias often small)
    for (int i = 1; i < blobs.size(); ++i) {
      CHECK_EQ(blobs[i - 1]->gpu_diff() + blobs[i - 1]->count(),
               blobs[i + 0]->gpu_diff());
    }
  #endif
    if (blobs.size() > 0) {
      // Make sure default stream is done computing gradients. Could be
      // replaced by cudaEventRecord+cudaStreamWaitEvent to avoid
      // blocking the default stream, but it's actually slower.
      CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));

      // Reduce asynchronously
      int size = 0;
      for (int i = 0; i < blobs.size(); ++i) {
        size += blobs[i]->count();
      }
      if (barrier_) {  // NULL in multi process case
        barrier_->wait();
      }
      NCCL_CHECK(ncclAllReduce(blobs[0]->mutable_gpu_diff(),
                               blobs[0]->mutable_gpu_diff(),
                               size,
                               nccl::dataType<Dtype>::type,
                               ncclSum, comm_, stream_));
      caffe_gpu_scal(size, (Dtype) 1.0 / Caffe::solver_count(),
                     blobs[0]->mutable_gpu_diff(), stream_);
    }
  }

  template<typename Dtype>
  void NCCL<Dtype>::on_gradients_ready() {
    if (solver_->param().layer_wise_reduce()) {
      CHECK_EQ(solver_->net()->params().size(),
               solver_->net()->learnable_params().size())
        << "Layer-wise reduce is not supported for nets with shared weights.";

      // Make sure reduction is done before applying gradients
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    } else {
      if (barrier_) {  // NULL in multi process case
        barrier_->wait();
      }
      NCCL_CHECK(ncclAllReduce(diff_, diff_, static_cast<int>(size_),
                               nccl::dataType<Dtype>::type, ncclSum, comm_,
                               cudaStreamDefault));
      caffe_gpu_scal(static_cast<int>(size_),
                     (Dtype) 1.0 / Caffe::solver_count(), diff_);
    }
  }

  template<typename Dtype>
  class Worker : public InternalThread {
   public:
    explicit Worker(shared_ptr<Solver<Dtype> > rank0, int device,
                    boost::barrier* barrier, vector<NCCL<Dtype>*>* nccls,
                    const char* restore)
      : rank0_(rank0), device_(device), barrier_(barrier),
        nccls_(nccls), restore_(restore) {
    }
    virtual ~Worker() {}

   protected:
    void InternalThreadEntry() {
      // Create solver and install callbacks
      SolverParameter param(rank0_->param());
      param.set_device_id(device_);
  #ifdef DEBUG
      int device;
      CUDA_CHECK(cudaGetDevice(&device));
      CHECK_EQ(device, device_);
  #endif
      param.set_type(rank0_->type());
      shared_ptr<Solver<Dtype> > s(SolverRegistry<Dtype>::CreateSolver(param));
      CHECK_EQ(s->type(), rank0_->type());
      if (restore_) {
        // Could not make NCCL broadcast solver state, it seems to crash
        // if called in a tight loop, regardless of barriers etc. so
        // restore all solvers from file.
        s->Restore(restore_);
      }
      NCCL<Dtype> nccl(s);
      nccl.set_barrier(barrier_);
      s->add_callback(&nccl);
      if (s->param().layer_wise_reduce()) {
        s->net()->add_after_backward(&nccl);
      }
      (*nccls_)[Caffe::solver_rank()] = &nccl;
      // Wait for other threads
      barrier_->wait();
      // Wait for NCCL init
      barrier_->wait();
      // Broadcast rank 0 state
      nccl.Broadcast();
      // Solve
      s->Step(param.max_iter() - s->iter());
      barrier_->wait();
  #ifdef DEBUG
      // Check all solvers have same state
      SGDSolver<Dtype>* sa = static_cast<SGDSolver<Dtype>*>(rank0_.get());
      SGDSolver<Dtype>* sb = static_cast<SGDSolver<Dtype>*>(s.get());
      for (int h = 0; h < sa->history().size(); ++h) {
        CUDA_CHECK(cudaSetDevice(sa->param().device_id()));
        const Dtype* a = sa->history()[h]->cpu_data();
        CUDA_CHECK(cudaSetDevice(sb->param().device_id()));
        const Dtype* b = sb->history()[h]->cpu_data();
        for (int v = 0; v < sa->history()[h]->count(); ++v) {
          CHECK_DOUBLE_EQ(a[v], b[v]);
        }
      }
  #endif
    }

    shared_ptr<Solver<Dtype> > rank0_;
    int device_;
    boost::barrier* barrier_;
    vector<NCCL<Dtype>*>* nccls_;
    const char* restore_;
  };

  template<typename Dtype>
  void NCCL<Dtype>::Run(const vector<int>& gpus, const char* restore) {
    boost::barrier barrier(static_cast<int>(gpus.size()));
    vector<NCCL<Dtype>*> nccls(gpus.size());
    // Create workers
    vector<shared_ptr<Worker<Dtype> > > workers(gpus.size());
    for (int i = 1; i < gpus.size(); ++i) {
      CUDA_CHECK(cudaSetDevice(gpus[i]));
      Caffe::set_solver_rank(i);
      Worker<Dtype>* w = new Worker<Dtype>(solver_, gpus[i], &barrier,
                                           &nccls, restore);
      w->StartInternalThread();
      workers[i].reset(w);
    }
    CUDA_CHECK(cudaSetDevice(gpus[0]));
    Caffe::set_solver_rank(0);
    barrier_ = &barrier;
    solver_->add_callback(this);
    if (solver_->param().layer_wise_reduce()) {
      solver_->net()->add_after_backward(this);
    }
    nccls[0] = this;
    // Wait for workers
    barrier.wait();
    // Init NCCL
    InitSingleProcess(&nccls);
    barrier.wait();
    // Run first solver on current thread
    Broadcast();
    solver_->Solve();
    barrier.wait();  // Hangs without it when running tests
    // Wait for shutdown
    for (int i = 1; i < gpus.size(); ++i) {
      workers[i]->StopInternalThread();
    }
  }

  INSTANTIATE_CLASS(GPUParams);
  INSTANTIATE_CLASS(Worker);
  INSTANTIATE_CLASS(NCCL);

#endif  // USE_NCCL
}  // namespace caffe
