#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::min;
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
        "The backend {leveldb, lmdb} containing the images");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Compute the mean_image of a set of images given by"
        " a leveldb/lmdb\n"
        "Usage:\n"
        "    compute_image_mean [FLAGS] INPUT_DB [OUTPUT_FILE]\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 2 || argc > 8 ) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean");
    return 1;
  }
  if (argc <= 3) return 1;
  int frame_num = 0;
  if( argc >= 3 )
 	frame_num = atoi(argv[2]); 
  CHECK_LE(frame_num, 5) << "frame_num should be <= 5";
  frame_num = min(argc-3, frame_num);

  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[1], db::READ);
  scoped_ptr<db::Cursor> cursor(db->NewCursor());

  BlobProto sum_blob[5];
  int count = 0;
  // load first datum
  Datum datum;
  datum.ParseFromString(cursor->value());

  if (DecodeDatumNative(&datum)) {
    LOG(INFO) << "Decoding Datum";
  }
  for ( int i = 0; i < frame_num; i++){
	  sum_blob[i].set_num(1);
	  sum_blob[i].set_channels(datum.channels());
	  sum_blob[i].set_height(datum.height());
	  sum_blob[i].set_width(datum.width());
  }	  
  const int data_size = datum.channels() * datum.height() * datum.width();
  int size_in_datum = std::max<int>(datum.data().size(),
                                    datum.float_data_size());
  
  
  for (int i = 0; i < size_in_datum; ++i) {
	for( int j=0; j<frame_num; ++j)    
		sum_blob[j].add_data(0.);
  }
  LOG(INFO) << "Starting Iteration";
  while (cursor->valid()) {
    Datum datum;
    datum.ParseFromString(cursor->value());
    DecodeDatumNative(&datum);

    const std::string& data = datum.data();
    size_in_datum = std::max<int>(datum.data().size(),
        datum.float_data_size());
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum;
    if (data.size() != 0) {
      CHECK_EQ(data.size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob[count%frame_num].set_data(i, sum_blob[count%frame_num].data(i) + (uint8_t)data[i]);
      }
    } else {
      CHECK_EQ(datum.float_data_size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
		sum_blob[count%frame_num].set_data(i, sum_blob[count%frame_num].data(i) + static_cast<float>(datum.float_data(i)));
		
      }
    }
    ++count;
    if (count % 10000 == 0) {
      LOG(INFO) << "Processed " << count << " files.";
    }
    cursor->Next();
  }

  if (count % 10000 != 0) {
    LOG(INFO) << "Processed " << count << " files.";
  }
  // Write to disk
  for(int j=0; j<frame_num; ++j) {
    for (int i = 0; i < sum_blob[j].data_size(); ++i) {
	  sum_blob[j].set_data(i, sum_blob[j].data(i)*5 / count);
    }
	LOG(INFO) << "Write to " << argv[3+j];
    WriteProtoToBinaryFile(sum_blob[j], argv[3+j]);
  }
  for(int j=0;j<frame_num;j++) {
	  const int channels = sum_blob[j].channels();
	  const int dim = sum_blob[j].height() * sum_blob[j].width();
	  std::vector<float> mean_values(channels, 0.0);
	  LOG(INFO) << "Number of channels: " << channels;
	  for (int c = 0; c < channels; ++c) {
		for (int i = 0; i < dim; ++i) {
		  mean_values[c] += sum_blob[j].data(dim * c + i);
		}
		LOG(INFO) << "mean_value frame [" << j << "] channel [" << c << "]:" << mean_values[c] / dim;
	  }
  }
  return 0;
}
