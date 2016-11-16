#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * BaseDataLayer作为DataLayer的基类，只有基本的函数
   BasePrefetchingDataLayer继承自BaseDataLayer和InternalThread，包含独立的读取数据线程
   DataLayer继承自BasePrefetchingDataLayer，是实际使用的layer
 */
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  // 初始化函数，只负责调用父类初始化和初始化内部成员transform_param_
  explicit BaseDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  // 所有Layer子类都有的初始化函数，此处会设置内部成员data_transformer_和output_labels_，
  // 用transform_param_初始化DataTransformer，并调用DataLayerSetUp()，
  // 在BasePrefetchingDataLayer中的LayerSetUp也会调用父类的这个函数
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  // 并行时，DataLayer应该共享，返回true
  virtual inline bool ShareInParallel() const { return true; }
  // 其他函数都没有实现
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected:
  // 保存DataTransformer的参数和实例
  TransformationParameter transform_param_;
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  bool output_labels_; // 判断是否要同时输出label blob
};

template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;
};

// 包含数据预取线程的数据层基类，使用两个BlockingQueue实现取数据线程和读数据的主线程的数据交换
// 会被DataLayer继承
template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  // 调用BaseDataLayer的初始化，初始化成员变量中的两个BlockingQueue
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  // 调用BaseDataLayer的LayerSetUp()函数，初始化成员变量prefetch_中的blobs，然后
  // 初始化DataTransformer并启动线程StartInternalThread()
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;

 protected:
  // 内部线程的工作函数，通过两个BlockingQueue，不断的将块从prefetch_free_中取出
  // 写入数据后放到prefetch_full_中
  virtual void InternalThreadEntry();
  // 实际读取一个数据块的函数
  virtual void load_batch(Batch<Dtype>* batch) = 0;

  // 存储预读取数据块的数据结构
  Batch<Dtype> prefetch_[PREFETCH_COUNT];
  BlockingQueue<Batch<Dtype>*> prefetch_free_;
  BlockingQueue<Batch<Dtype>*> prefetch_full_;

  Blob<Dtype> transformed_data_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
