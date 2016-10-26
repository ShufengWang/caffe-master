#include "caffe/util/db.hpp"
#include "caffe/util/db_leveldb.hpp"
#include "caffe/util/db_lmdb.hpp"

#include <string>

namespace caffe { namespace db {

DB* GetDB(DataParameter::DB backend) {
  switch (backend) {
  case DataParameter_DB_LEVELDB:
    return new LevelDB();
  case DataParameter_DB_LMDB:
    return new LMDB();
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}



DB* GetDB(const string& backend) {
  if (backend == "leveldb") {
    return new LevelDB();
  } else if (backend == "lmdb") {
    return new LMDB();
  } else {
    LOG(FATAL) << "Unknown database backend";
  }
}

// added by wsf at 2015.10.14
DB* GetDB(FileDataParameter::DB backend) {
  switch (backend) {
  case FileDataParameter_DB_LEVELDB:
    return new LevelDB();
  case FileDataParameter_DB_LMDB:
    return new LMDB();
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

// added by wsf at 2015.10.15
DB* GetDB(MultiChannelsDataParameter::DB backend) {
  switch (backend) {
  case MultiChannelsDataParameter_DB_LEVELDB:
    return new LevelDB();
  case MultiChannelsDataParameter_DB_LMDB:
    return new LMDB();
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}


}  // namespace db
}  // namespace caffe
