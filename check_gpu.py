import os, importlib.util
spec = importlib.util.find_spec("nvidia")
root = list(spec.submodule_search_locations)[0]
for sub in os.listdir(root):
    b = os.path.join(root, sub, "bin")
    if os.path.isdir(b):
        os.add_dll_directory(b)
        os.environ["PATH"] = b + os.pathsep + os.environ["PATH"]

import tensorflow as tf
print("TF", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
if gpus:
    a = tf.random.normal((1000, 1000))
    print("matmul ok:", float(tf.reduce_sum(tf.matmul(a, a))))
