#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "caffe/common.hpp"
#include "caffe/util/insert_splits.hpp"

namespace caffe {

// 如果一个层的top blob会被很多个层用到（作bottom blob或loss），则会在这个top blob后增加一个split层
// 这个split层读取这个top blob并输出多个blob，每个输出的blob分别作为上面多个层的bottom blob
void InsertSplits(const NetParameter& param, NetParameter* param_split) {
  // Initialize by copying from the input NetParameter.
  param_split->CopyFrom(param);
  param_split->clear_layer();
  map<string, pair<int, int> > blob_name_to_last_top_idx; // blob名称对应第i层的第j个top blob
  // 第i层第j个bottom blob对应第i2层第j2个top blob
  map<pair<int, int>, pair<int, int> > bottom_idx_to_source_top_idx;
  // 作为top blob的第i层第j个blob，被使用（作为其他层的bottom blob或作为loss且loss_weight非0）的次数
  map<pair<int, int>, int> top_idx_to_bottom_count;
  // top blob作为loss时的loss_weight值, 为0时也保存
  map<pair<int, int>, float> top_idx_to_loss_weight;
  // top blob需要被split时，存储已split的次数。split时会用这个次数作为新top blob的名称的一部分
  // C++ map用[]取访问时，如果key不存在，会增加key并设置默认value值
  map<pair<int, int>, int> top_idx_to_bottom_split_idx;
  // layer id到layer名称的映射
  map<int, string> layer_idx_to_layer_name;
  // 遍历一遍layers，建立top blob和bottom blob的对应关系，统计top blob被用到的计数，保存top blob作为
  // loss时的loss_weight
  for (int i = 0; i < param.layer_size(); ++i) {
    const LayerParameter& layer_param = param.layer(i);
    layer_idx_to_layer_name[i] = layer_param.name();
    for (int j = 0; j < layer_param.bottom_size(); ++j) {
      const string& blob_name = layer_param.bottom(j);
      // 通过bottom blob的名称，查找对应的top blob的位置
      if (blob_name_to_last_top_idx.find(blob_name) ==
          blob_name_to_last_top_idx.end()) {
        LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
                   << layer_param.name() << "', bottom index " << j << ")";
      }
      const pair<int, int>& bottom_idx = make_pair(i, j);
      const pair<int, int>& top_idx = blob_name_to_last_top_idx[blob_name];
      bottom_idx_to_source_top_idx[bottom_idx] = top_idx;
      ++top_idx_to_bottom_count[top_idx];
    }

    // 更新top blob的名称位置字典
    for (int j = 0; j < layer_param.top_size(); ++j) {
      const string& blob_name = layer_param.top(j);
      blob_name_to_last_top_idx[blob_name] = make_pair(i, j);
    }
    // A use of a top blob as a loss should be handled similarly to the use of
    // a top blob as a bottom blob to another layer.
    // 如果top blob作为loss也要像作为bottom blob一样被split，所以也要计数
    const int last_loss =
        std::min(layer_param.loss_weight_size(), layer_param.top_size());
    for (int j = 0; j < last_loss; ++j) {
      const string& blob_name = layer_param.top(j);
      const pair<int, int>& top_idx = blob_name_to_last_top_idx[blob_name];
      top_idx_to_loss_weight[top_idx] = layer_param.loss_weight(j);
      if (top_idx_to_loss_weight[top_idx]) {
        ++top_idx_to_bottom_count[top_idx];
      }
    }
  }
  // 遍历第二遍layers，添加split layer并重新设置top blob和bottom blob的对应关系
  for (int i = 0; i < param.layer_size(); ++i) {
    LayerParameter* layer_param = param_split->add_layer();
    layer_param->CopyFrom(param.layer(i));
    // Replace any shared bottom blobs with split layer outputs.
    // 需要的话，将bottom blob设置为split后的top blob
    for (int j = 0; j < layer_param->bottom_size(); ++j) {
      const pair<int, int>& top_idx =
          bottom_idx_to_source_top_idx[make_pair(i, j)];
      const int split_count = top_idx_to_bottom_count[top_idx];
      if (split_count > 1) {
        const string& layer_name = layer_idx_to_layer_name[top_idx.first];
        const string& blob_name = layer_param->bottom(j);
        layer_param->set_bottom(j, SplitBlobName(layer_name,
            blob_name, top_idx.second, top_idx_to_bottom_split_idx[top_idx]++));
      }
    }
    // Create split layer for any top blobs used by other layer as bottom
    // blobs more than once.
    // 对所有的top blob，根据需要建立split layer
    for (int j = 0; j < layer_param->top_size(); ++j) {
      const pair<int, int>& top_idx = make_pair(i, j);
      const int split_count = top_idx_to_bottom_count[top_idx];
      if (split_count > 1) {
        const string& layer_name = layer_idx_to_layer_name[i];
        const string& blob_name = layer_param->top(j);
        LayerParameter* split_layer_param = param_split->add_layer();
        const float loss_weight = top_idx_to_loss_weight[top_idx];
        ConfigureSplitLayer(layer_name, blob_name, j, split_count,
            loss_weight, split_layer_param);
        if (loss_weight) {
          // layer_param已经添加到split_layer_param里了，所以这里清空
          layer_param->clear_loss_weight();
          // 如果top blob作为loss，因为在遍历上面层的bottom blob时不会累加计数，所以要在这里增加一次
          top_idx_to_bottom_split_idx[top_idx]++;
        }
      }
    }
  }
}

// 新增split 层，将参数保存到split_layer_param中
void ConfigureSplitLayer(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_count, const float loss_weight,
    LayerParameter* split_layer_param) {
  split_layer_param->Clear();
  split_layer_param->add_bottom(blob_name);
  split_layer_param->set_name(SplitLayerName(layer_name, blob_name, blob_idx));
  split_layer_param->set_type("Split");
  for (int k = 0; k < split_count; ++k) {
    split_layer_param->add_top(
        SplitBlobName(layer_name, blob_name, blob_idx, k));
    if (loss_weight) {
      if (k == 0) {
        // 将原top，现在是bottom的blob的loss_weight的设置为新层的第一个top blob的loss_weight
        split_layer_param->add_loss_weight(loss_weight);
      } else {
        split_layer_param->add_loss_weight(0);
      }
    }
  }
}

// 新增的split层的名称，比如conv1_convlayer1_0_split
// 这个层的bottom是convlayer1层的第0个blob，conv1，
// 这个层的top是conv1_convlayer1_0_split_0,conv1_convlayer1_0_split_1,...
string SplitLayerName(const string& layer_name, const string& blob_name,
    const int blob_idx) {
  ostringstream split_layer_name;
  split_layer_name << blob_name << "_" << layer_name << "_" << blob_idx
      << "_split";
  return split_layer_name.str();
}

// split后的top blob的名称，比如conv1_convlayer1_0_split_2
string SplitBlobName(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_idx) {
  ostringstream split_blob_name;
  split_blob_name << blob_name << "_" << layer_name << "_" << blob_idx
      << "_split_" << split_idx;
  return split_blob_name.str();
}

}  // namespace caffe
