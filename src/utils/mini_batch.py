import random


def generate_list_of_mini_batches(input_data, batch_size):
    """
    This function is to generate a list of mini-batches of size "batch_size" based on the input data
    Args:
        input_data (list): input dataset
        batch_size (int): maximal size of a mini-batch, the last mini-batch's size might be less than the batch_size

    Returns:
        list: list of mini-batches

    """

    data_size = len(input_data)

    # Shuffle data to get better training performance
    # we can use random.shuffle() to shuffle a list, however, it is a in-place function
    shuffled_data = random.sample(input_data, data_size)

    # Create a list of mini-batches
    mini_batch_list = [shuffled_data[i:(i + batch_size)] for i in range(0, data_size, batch_size)]

    return mini_batch_list
