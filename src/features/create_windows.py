import tensorflow as tf

window_size = 3
def windowed_dataset(series, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(series.reshape(-1, 1))

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # Create tuples with features and labels
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    # Shuffle the windows
    # dataset = dataset.shuffle(shuffle_buffer)

    # Create batches of windows
    dataset = dataset.batch(len(series)).prefetch(1)

    return dataset