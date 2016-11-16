#ifndef CAFFE_DATA_READER_HPP_
#define CAFFE_DATA_READER_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * @brief Reads data from a source to queues available to data layers.
 * A single reading thread is created per source, even if multiple solvers
 * are running in parallel, e.g. for multi-GPU training. This makes sure
 * databases are read sequentially, and that each solver accesses a different
 * subset of the database. Data is distributed to solvers in a round-robin
 * way to keep parallel training deterministic.
 */
// 从共享的资源读取数据然后排队输入到数据层，每个资源创建单个线程，即便是使用多
// 个GPU在并行任务中求解。这就保证对于频繁读取数据库，并且每个求解的线程使用的
// 子数据是不同的。数据成功设计就是这样使在求解时数据保持一种循环地并行训练
class DataReader {
 public:
  explicit DataReader(const LayerParameter& param);
  ~DataReader();

  inline BlockingQueue<Datum*>& free() const {
    return queue_pair_->free_;
  }
  inline BlockingQueue<Datum*>& full() const {
    return queue_pair_->full_;
  }

 protected:
  // Queue pairs are shared between a body and its readers
  // 在reader_和它的调用者之间使用两个BlockingQueue使从数据库读数据和将
  // 读到的数据给调用者能多线程同时进行
  // 初始化size个Datum，这些Datum会在free_和full_两个Queue中移动
  class QueuePair {
   public:
    explicit QueuePair(int size);
    ~QueuePair();

    BlockingQueue<Datum*> free_;
    BlockingQueue<Datum*> full_;

  DISABLE_COPY_AND_ASSIGN(QueuePair);
  };

  // A single body is created per source
  // 实际的从数据库读取数据的线程
  class Body : public InternalThread {
   public:
    explicit Body(const LayerParameter& param);
    virtual ~Body();

   protected:
    void InternalThreadEntry();
    void read_one(db::Cursor* cursor, QueuePair* qp);

    const LayerParameter param_;
    // 保存DataReader中存储数据的QueuePair
    BlockingQueue<shared_ptr<QueuePair> > new_queue_pairs_;

    // 设置为友元
    friend class DataReader;

  DISABLE_COPY_AND_ASSIGN(Body);
  };

  // A source is uniquely identified by its layer name + path, in case
  // the same database is read from two different locations in the net.
  // 用layer name和读取数据路径来标注引用这个DataReader的调用者
  static inline string source_key(const LayerParameter& param) {
    return param.name() + ":" + param.data_param().source();
  }

  // 内部成员，queuepair用于保存读取的数据，body_用于执行从数据库读取数据的操作
  const shared_ptr<QueuePair> queue_pair_;
  shared_ptr<Body> body_;

  // 使用静态map成员保存所有正在运行的读取数据线程对象
  static map<const string, boost::weak_ptr<DataReader::Body> > bodies_;

DISABLE_COPY_AND_ASSIGN(DataReader);
};

}  // namespace caffe

#endif  // CAFFE_DATA_READER_HPP_
