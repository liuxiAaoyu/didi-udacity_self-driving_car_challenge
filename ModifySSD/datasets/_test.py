import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


def _EncodedFloatFeature(ndarray):
    return tf.train.Feature(float_list=tf.train.FloatList(
        value=ndarray.flatten().tolist()))


def _EncodedInt64Feature(ndarray):
    return tf.train.Feature(int64_list=tf.train.Int64List(
        value=ndarray.flatten().tolist()))


def _GetNdArray(a):
    if not isinstance(a, np.ndarray):
      a = np.array(a)
    return a


def assertAllEqual(a, b):
    """Asserts that two numpy arrays have the same values.

    Args:
      a: a numpy ndarray or anything can be converted to one.
      b: a numpy ndarray or anything can be converted to one.
    """
    a = _GetNdArray(a)
    b = _GetNdArray(b)

    same = (a == b)

    if a.dtype == np.float32 or a.dtype == np.float64:
      same = np.logical_or(same, np.logical_and(np.isnan(a), np.isnan(b)))
    if not np.all(same):
      # Prints more details than np.testing.assert_array_equal.
      diff = np.logical_not(same)
      if a.ndim:
        x = a[np.where(diff)]
        y = b[np.where(diff)]
        print("not equal where = ", np.where(diff))
      else:
        # np.where is broken for scalars
        x, y = a, b
      print("not equal lhs = ", x)
      print("not equal rhs = ", y)
      np.testing.assert_array_equal(a, b)


def tfrecord_test(sess):
    np_image = np.random.rand(2, 2, 11).astype('f')

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _EncodedFloatFeature(np_image),
        'image/shape': _EncodedInt64Feature(np.array(np_image.shape)),
    }))

    serialized_example = example.SerializeToString()

    with sess.as_default():
        serialized_example = tf.reshape(serialized_example, shape=[])
        keys_to_features = {
            'image': tf.VarLenFeature(dtype=tf.float32),
            'image/shape': tf.VarLenFeature(dtype=tf.int64),
        }
        items_to_handlers = {
            'image':
                slim.tfexample_decoder.Tensor(
                    'image', shape_keys='image/shape'),
        }
        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                        items_to_handlers)
        [tf_image] = decoder.decode(serialized_example,
                                    ['image'])
        assertAllEqual(tf_image.eval(), np_image)
        print(tf_image.eval())
        print(np_image.shape)
        print(tf_image.get_shape().ndims)
        print(tf.shape(tf_image).eval())
        print("ok")

        t1, t2, t3, t4 = tf.split(tf_image, [1, 1, 1, 8], axis=2)
        scale = tf.constant(256.0)
        shift = tf.constant(0.5)
        t1 = tf.subtract(tf.divide(t1, scale), shift)
        scale = tf.constant(32.0)
        t2 = tf.subtract(tf.divide(t2, scale), shift)
        scale = tf.constant(50.0)
        t3 = tf.subtract(tf.divide(t3, scale), shift)

        scale = tf.constant(160.0)
        t4 = tf.subtract(tf.divide(t4, scale), shift)

        t = tf.concat([t1, t2, t3, t4], axis=2)
        #assertAllEqual(t.eval(), np_image)
        print(t.eval())


with tf.Session() as sess:
    tfrecord_test(sess)
