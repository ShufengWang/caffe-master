#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
MultiChannelsDataLayer<Dtype>::~MultiChannelsDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void MultiChannelsDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Initialize DB
  db_.reset(db::GetDB(this->layer_param_.multi_channels_data_param().backend()));
  db_->Open(this->layer_param_.multi_channels_data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());

  // Check if we should randomly skip a few data points
  if (this->layer_param_.multi_channels_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.multi_channels_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      cursor_->Next();
    }
  }

  // added by wsf at 2015.10.15
  const int frame_num = this->layer_param_.multi_channels_data_param().frame_num();

  // Read a data point, to initialize the prefetch and top blobs.
  Datum datum;
  datum.ParseFromString(cursor_->value());
	
  // added by wsf at 2015.10.11
  LOG(INFO) << "output datum size: "
      << datum.channels() << "," << datum.height() << ","
      << datum.width() << "," << datum.encoded();

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  top_shape[1] = datum.channels()*frame_num;

  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = this->layer_param_.multi_channels_data_param().batch_size();
  this->prefetch_data_.Reshape(top_shape);
  top[0]->ReshapeLike(this->prefetch_data_);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, this->layer_param_.multi_channels_data_param().batch_size());
    top[1]->Reshape(label_shape);
    this->prefetch_label_.Reshape(label_shape);
  }
  // added by wsf at 2015.10.13
  //for( int i=0;i<10;i++){
  //	LOG(INFO) << "check blob data "<< (top[0]->cpu_data())[i];
  //	LOG(INFO) << "check blob data label "<< (top[1]->cpu_data())[i];
  //}
	
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void MultiChannelsDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.multi_channels_data_param().batch_size();
  // added by wsf at 2015.10.15
  const int frame_num = this->layer_param_.multi_channels_data_param().frame_num();
  

  Datum datum;
  datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  top_shape[1] = datum.channels()*frame_num;
  
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  this->prefetch_data_.Reshape(top_shape);

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  
  	

  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  
 

  timer.Start();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
	// added by wsf at 2015.10.19
	int offset = this->prefetch_data_.offset(item_id);
	this->transformed_data_.set_cpu_data(top_data + offset);
			
	
    // get several datum
    for (int frame_id = 0; frame_id < frame_num; ++frame_id) {
		// get a datum
		Datum datum;
		datum.ParseFromString(cursor_->value());
		
		read_time += timer.MicroSeconds();

		timer.Start();				
		// Apply data transformations (mirror, scale, crop...)
		//int offset = this->prefetch_data_.offset(item_id,datum.channels()*frame_id);
		//this->transformed_data_.set_cpu_data(top_data + offset);
		// 输入frame_id,因为不同的通道存到tranfromed_data_的指针也不一样，mean_file也不一样		
		this->data_transformer_->Transform(datum, &(this->transformed_data_), frame_id);
		
		// Copy label.
					
		if ( frame_id==0 && this->output_labels_) {
		  top_label[item_id] = datum.label();
		}
		trans_time += timer.MicroSeconds();
		timer.Start();
		
		cursor_->Next();
		if (!cursor_->valid()) {
      		DLOG(INFO) << "Restarting data prefetching from start.";
      		cursor_->SeekToFirst();
    	}
	}
	/*
	if(item_id==0){	
		// added by wsf at 2015.10.13
		LOG(INFO) << "check blob data label "<< top_label[0];
		LOG(INFO) << "check blob data ";

		Dtype* my_data = this->transformed_data_.mutable_cpu_data();

		for( int i=0;i<3*260*260;i++){
		  LOG(INFO) << top_data[i] << " " << my_data[i];
					
		}
	}
	*/
	
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
  
  
  
	
}

INSTANTIATE_CLASS(MultiChannelsDataLayer);
REGISTER_LAYER_CLASS(MultiChannelsData);

}  // namespace caffe
