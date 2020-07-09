def test_search_space():
    from random import random

    from tensorflow.keras.utils import plot_model

    import deephyper.search.nas.model.tensorflow1x as tfm

    struct = tfm.baseline.dense_skipco.create_search_space()

    ops = [random() for _ in range(struct.num_nodes)]
    struct.set_ops(ops)
    model = struct.create_model()

    plot_model(model, to_file=f"test_dense_skipco.png", show_shapes=True)
