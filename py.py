import tensorflow as tf

# Load Keras .h5 model
model = tf.keras.models.load_model("model.h5")

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (Optional) Optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert model
tflite_model = converter.convert()

# Save TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model converted to TFLite successfully!")
