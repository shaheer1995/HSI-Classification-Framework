import numpy as np

def sampling(proportion, ground_truth):
  train = {}
  test = {}
  labels_loc = {}
  m = max(ground_truth)
  for i in range(m):
      indexes = [
          j for j, x in enumerate(ground_truth.ravel().tolist())
          if x == i + 1
      ]
      # print(len(indexes))
      np.random.shuffle(indexes)
      labels_loc[i] = indexes
      if proportion != 1:
          nb_val = max(int((1 - proportion) * len(indexes)), 3)
          # print(len(indexes),nb_val)
      else:
          nb_val = 0
      train[i] = indexes[:200]
      # if nb_val>3000:
      #   test[i] = indexes[nb_val:6000]
      # elif nb_val>2000:
      #   test[i] = indexes[nb_val:5000]
      # else:
      test[i] = indexes[200:1200]
      print('C',i+1,'\tTotal : ', len(indexes),'\ttrain : ',len(train[i]),'\ttest : ',len(test[i]))
  train_indexes = []
  test_indexes = []
  for i in range(m):
      train_indexes += train[i]
      test_indexes += test[i]
  np.random.shuffle(train_indexes)
  np.random.shuffle(test_indexes)

  print('Total Train : ', len(train_indexes))
  print('Total Test : ',len(test_indexes))

  return train_indexes, test_indexes


def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1)]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch


def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    return small_cubic_data