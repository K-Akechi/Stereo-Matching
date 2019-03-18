import tensorflow as tf


flags = tf.app.flags

flags.DEFINE_integer('batch_size', 8, 'Batch size')
flags.DEFINE_integer('epochs', 10, 'Num of epochs')
flags.DEFINE_string('model', 'basic', 'Types of model')
flags.DEFINE_string('datapath', 'dataset/', 'datasets')
flags.DEFINE_string('model_dir', 'model', 'Trained network')
flags.DEFINE_string('load_model_dir', 'model', 'Load pretrained network')
