import tensorflow as tf

print("Bienvenida al cerebro de CineScope")
print("TensorFlow:", tf.__version__)

numero = tf.constant(10)

print("Número almacenado:", numero.numpy())