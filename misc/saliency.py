
import tensorflow as tf

def compute_gradcam_maps(sess, X, model, task='rotation', layer=None):
	"""
	Compute a class saliency map using the model for images X and labels y.

	Input:
	- X: Input images, numpy array of shape (N, H, W, 3)
	- model: A SqueezeNet model that will be used to compute the saliency map.
	- task: EfficientPose task: rotation, translation or classification
	- layer: Name of the convolutional layer to be used for gradient computation
	Returns:
	- saliency: A numpy array of shape (N, H, W) giving the saliency maps for the
	input images.
	"""

	assert task in ["rotation", "translation"]
	layer0 = f'{task}_net/{task}-0'
	layer1 = f'{task}_net/{task}-1'
	layer2 = f'{task}_net/{task}-2'

	# Compute the score of the correct class for each example.
	# This gives a Tensor with shape [N], the number of examples.
	boxes, scores, labels, rotations, translations = model.outputs

	conv0_out = model.get_layer(layer0).output
	conv1_out = model.get_layer(layer1).output
	conv2_out = model.get_layer(layer2).output

	conv_layer_output = tf.concat([conv0_out, conv1_out, conv2_out], axis=3)

	# Select model output
	best_pred = tf.argmax(scores, axis=1)

	# Select output for the right task
	if task == "rotation":
		correct_scores = tf.gather(rotations, indices=best_pred, axis=1)
	else:
		correct_scores = tf.gather(translations, indices=best_pred, axis=1)

	# gradient computation based on model output and model image:
	grads_input_0 = tf.gradients(correct_scores, conv0_out)[0]
	grads_input_1 = tf.gradients(correct_scores, conv1_out)[0]
	grads_input_2 = tf.gradients(correct_scores, conv2_out)[0]

	grads_input = tf.concat([grads_input_0, grads_input_1, grads_input_2], axis=3)
	#pooled_grads = tf.math.reduce_mean(tf.abs(grads_input), axis=(1, 2))
	pooled_grads = tf.linalg.norm(grads_input, axis=(1, 2))

	heatmap = conv_layer_output * tf.expand_dims(tf.expand_dims(pooled_grads, axis=1), 1)
	saliency = tf.math.reduce_sum(tf.abs(heatmap), reduction_indices=[3])

	# saliency computation:
	feed_dict = {model.inputs[0]: X[0], model.inputs[1]: X[1]}  # , model.labels: y}
	saliency = sess.run(saliency, feed_dict=feed_dict)

	return saliency

def compute_backprop_maps(sess, X, model, task='rotation'):
	"""
	Compute a class saliency map using the model for images X and labels y.

	Input:
	- X: Input images, numpy array of shape (N, H, W, 3)
	- model: A SqueezeNet model that will be used to compute the saliency map.

	Returns:
	- saliency: A numpy array of shape (N, H, W) giving the saliency maps for the
	input images.
	"""

	# Compute the score of the correct class for each example.
	# This gives a Tensor with shape [N], the number of examples.
	boxes, scores, labels, rotations, translations = model.outputs

	best_pred = tf.argmax(scores, axis=1)
	# Select output for the right task
	if task == "rotation":
		correct_scores = tf.gather(rotations, indices=best_pred, axis=1)
	else:
		correct_scores = tf.gather(translations, indices=best_pred, axis=1)

	#gradient computation based on model output and model image:
	grads_input = tf.gradients(correct_scores, model.inputs[0])[0]

	# Take the maximum value over the 3 input channels.
	saliency = tf.math.reduce_max(grads_input, reduction_indices=[3])

	# All entries are nonnegative.
	# saliency = tf.cast(saliency > 0, saliency.dtype) * saliency

	# saliency computation:
	feed_dict = {model.inputs[0]: X[0], model.inputs[1]: X[1]}  # , model.labels: y}
	saliency = sess.run(saliency, feed_dict=feed_dict)

	return saliency
