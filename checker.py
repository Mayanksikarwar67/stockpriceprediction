import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow Version:", tf.__version__)
print("Is GPU available?", tf.test.is_gpu_available())
print("Available devices:", tf.config.list_physical_devices())
with tf.device('/GPU:0'):
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[checkpoint, reduce_lr])
