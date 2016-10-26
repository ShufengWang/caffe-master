addpath('matlab');

model = './models/mycaffenet/train_val.prototxt';
%weights = './models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';

caffe.set_mode_gpu();

net = caffe.Net(model, 'test');

solver = caffe.Solver('./models/mycaffenet/solver.prototxt');

solver.solve();

