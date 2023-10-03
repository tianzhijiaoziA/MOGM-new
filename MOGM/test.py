# -*- coding: UTF-8 -*- 
# 测试 TensorFlow 是否正常

try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    with tf.compat.v1.Session() as sess:
        devices = sess.list_devices()
        print("Available devices:")
        for device in devices:
            print(device.name)
except ImportError:
    print("TensorFlow is not installed")

# 测试 PyTorch 是否正常
try:
    import torch
    print("PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
    else:
        print("CUDA is not available")
except ImportError:
    print("PyTorch is not installed")
