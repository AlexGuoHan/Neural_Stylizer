import vgg
import tensorflow as tf
import numpy as np
from sys import stderr
from datetime import datetime

CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

try:
	reduce
except NameError:
	from functools import reduce


def stylize(network, initial, content, styles, iterations, content_weight, style_weight, style_blend_weights, tv_weight, learning_rate, print_iterations=None, checkpoint_iterations=None):

	# input.shape = (n_image, height, width, channel)
	content_shape = (1, ) + content.shape
	# style.shape = [ all style shapes ]
	style_shapes = [(1, ) + style.shape for style in styles]
	content_features = {}
	style_features = [{} for _ in styles] # for multiple style image inputs

	# compute content features
	g = tf.Graph()
	with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
		content_pl = tf.placeholder('float', shape=content_shape)
		# compute feedforward activations
		# net is the activation of image using Placeholder
		activation, mean_pixel = vgg.net(network, content_pl)

		# preprocessing the input
		content_preprocessed = np.array([vgg.preprocess(content, mean_pixel)])

		# extract content features using preprocessed input into the VGG
		# we only extract content features from one layer
		content_features[CONTENT_LAYER] = activation[CONTENT_LAYER].eval(
			feed_dict={content_pl: content_preprocessed})


	# compute style features
	# the loop below is for multiple style image inputs
	for i in range(len(styles)):
		g = tf.Graph()
		with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
			# since different style layers have differnt shapes
			style_pl = tf.placeholder('float', shape=style_shapes[i])
# question: why we use the mean value from content_features
			activation, _ = vgg.net(network, style_pl)
			style_preprocessed = np.array([vgg.preprocess(styles[i], mean_pixel)])
			# since we will compute multiple layers of styles, we use loop
			for layer in STYLE_LAYERS:
				# extract the one of the style features from one layer 
				_features = activation[layer].eval(feed_dict={style_pl: style_preprocessed})
				# since we will compute the Gram Matrix, we will reshape the output
				# so that the inner product is easier to compute
# question why should we reshape? what is the origal shape? what does -1 mean
				_features = _features.reshape((-1, _features.shape[3]))
				# compute the Gram Matrix as style features
# why divide it by _features.size
				gram = np.matmul(_features.T, _features) / _features.size
				style_features[i][layer] = gram  # the first index is the n_th style image input

	# compute back-prop
	with tf.Graph().as_default():
		# initial = None means this iteration is our first iteration
		# thus we need to generate a white noise image
		if initial is None:
# the noise turned out to be not used at all
			white_noise_image = np.random.normal(size=content_shape, scale=np.std(content) * .1)
			initial = tf.random_normal(content_shape) * .256
		# if we already have an image in training
		# we will keep using this image for further modification
		else:
			initial_preprocessed = np.array([vgg.preprocess(initial, mean_pixel)])
			initial = initial_preprocessed.astype('float32')
		# we make this initial input as a trainable variable
		image = tf.Variable(initial)
		activation, _ = vgg.net(network, image)

		# compute content loss
		image_content_features = activation[CONTENT_LAYER]
		target_content_features = content_features[CONTENT_LAYER]
# why divide it by target.size, can we eliminate that?
		# the content weight is included here rather than the end
		content_loss = content_weight * .5 * 1 / target_content_features.size * tf.nn.l2_loss(image_content_features - target_content_features)

		# compute style loss
		# using loop to sum style loss for multiple style image inputs
		style_loss_for_all_styles = 0
		for i in range(len(styles)):
			style_losses = []  # the total losses
			# using loop to sum style loss for multiple style layers
			for style_layer in STYLE_LAYERS:
				layer_activation = activation[style_layer]
				_, height, width, channel = map(
					lambda i: i.value, layer_activation.get_shape())
				layer_size = height * width * channel
				feats = tf.reshape(layer_activation, (-1, channel))
# it doesn't have to divide by size
				image_style_gram = tf.matmul(tf.transpose(feats), feats) / layer_size
				target_style_gram = style_features[i][style_layer]
				layer_style_loss = 2 / target_style_gram.size * tf.nn.l2_loss(image_style_gram - target_style_gram)
				style_losses.append(layer_style_loss)
			style_loss_for_all_styles += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)

		# total variation denoising
		# this loss is added to regularize that
		# the output image will not deviate too much
		# from content image at each pixel
		tv_y_size = _tensor_size(image[:, 1:, :, :])
		tv_x_size = _tensor_size(image[:, :, 1:, :])
		tv_loss = tv_weight * 2 * (
			(tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :content_shape[1] - 1, :, :]) / tv_y_size) +
			(tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :content_shape[2] - 1, :]) / tv_x_size))

		# overall loss
		loss = content_loss + style_loss_for_all_styles + tv_loss

		# optimizer
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

		def print_progress(i, last=False):
			
			stderr.write('Iteration %d/%d' % (i + 1, iterations))
			
			if last or (print_iterations and i % print_iterations == 0):
				stderr.write('  content loss: %g\n' % content_loss.eval())
				stderr.write('    style loss: %g\n' % style_loss_for_all_styles.eval())
				stderr.write('       tv loss: %g\n' % tv_loss.eval())
				stderr.write('    total loss: %g\n' % loss.eval())

		# optimization
		best_loss = float('inf')  # all losses will be lower than initial
		best = None
		total_initial_time = datetime.now().replace(microsecond=0)
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())
			initial_time = datetime.now().replace(microsecond=0)
			for i in range(iterations):
				now_time = datetime.now().replace(microsecond=0)
				last_step = (i == iterations - 1)
				print_progress(i, last=last_step)
				stderr.write(' Training Time %s  Elapsed Time %s\n' % (str(now_time - initial_time), str(now_time - total_initial_time)))
				initial_time = now_time
				train_step.run()

				# when checkpoint_iterations is not None
				# and when iter idx fulfills it
				# or when it comes to the last step
				if(checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
					this_loss = loss.eval()
					if this_loss < best_loss:
						best_loss = this_loss
						# image was a tf.Variable
						# eval it turns it into numbers, I guess
						best = image.eval()
# what is yield?
					# content_shape[1:] means 
					# shape[1], shape[2], shape[3]
					# but eliminate the shape[0] which 
					# means the number of images
					yield(
						(None if last_step else i),
						(vgg.unprocess(best.reshape(content_shape[1:]), mean_pixel)))


def _tensor_size(tensor):
	from operator import mul
	return reduce(mul, (d.value for d in tensor.get_shape()), 1)
