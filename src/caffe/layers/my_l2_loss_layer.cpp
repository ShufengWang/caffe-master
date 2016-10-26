#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MyL2LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[2]->count(), bottom[1]->count())
      << "weights must have the same dimension with data.";
  diff_.ReshapeLike(*bottom[0]);
  diff_square_.ReshapeLike(*bottom[0]);
  weight_diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void MyL2LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  
  caffe_mul(
	  count,
      diff_.cpu_data(),
	  diff_.cpu_data(),
	  diff_square_.mutable_cpu_data());
  // LOG(INFO) << "count: " << bottom[2]->count();
  // for(int i=100;i<110;i++)
  //	LOG(INFO) << "label[" <<i<<"]: " << bottom[1]->cpu_data()[i]
  //			  << " btm[" <<i<<"]: " << bottom[2]->cpu_data()[i];
  
  // 
  Dtype dot = caffe_cpu_dot(count, diff_square_.cpu_data(), bottom[2]->cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

  //LOG(INFO) << "loss: " << loss ;
}

template <typename Dtype>
void MyL2LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
	  
	  caffe_mul(bottom[2]->count(),
			bottom[2]->cpu_data(),diff_.cpu_data(), weight_diff_.mutable_cpu_data());
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          weight_diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MyL2LossLayer);
#endif

INSTANTIATE_CLASS(MyL2LossLayer);
REGISTER_LAYER_CLASS(MyL2Loss);

}  // namespace caffe
