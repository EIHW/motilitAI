import tensorflow
import gc
from tensorflow.compat.v1.keras.backend import set_session, clear_session, get_session


def reset_keras(model):
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del model  # this is from global space - change this as you need
    except:
        pass

    # if it's done something you should see a number being outputted
    print(gc.collect())

    # use the same config as you used to create the session
    config = tensorflow.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.compat.v1.Session(config=config))