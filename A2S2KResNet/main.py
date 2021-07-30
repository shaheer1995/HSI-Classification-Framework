import torch
import collections
import time

import numpy as np
import torch.optim as optim
from sklearn import metrics, preprocessing

import record
import torch_optimizer as optim2
import models
from torchsummary import summary
import torch.utils.data as Data
import dataloader
import utils
import train

PARAM_DATASET = 'WHU_HC'
PARAM_EPOCH = 100
PARAM_ITER = 3
PATCH_SIZE = 6
PARAM_VAL = 0.95
PARAM_OPTIM = 'adam'
PARAM_KERNEL_SIZE = 24

# # Data Loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for Monte Carlo runs
seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]
ensemble = 1

global Dataset  # WHU_HC, WHU_Hh, WHU_LK
dataset = PARAM_DATASET  # input('Please input the name of Dataset(IN, UP, SV, KSC):')
Dataset = dataset.upper()

# # Pytorch Data Loader Creation
data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT = dataloader.load_dataset(Dataset, PARAM_VAL)
print(data_hsi.shape)
image_x, image_y, BAND = data_hsi.shape
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )
CLASSES_NUM = max(gt)
print('The class numbers of the HSI data is:', CLASSES_NUM)

print('-----Importing Setting Parameters-----')
ITER = PARAM_ITER
PATCH_LENGTH = PATCH_SIZE
lr, num_epochs, batch_size = 0.001, 200, 16
loss = torch.nn.CrossEntropyLoss()

img_rows = 2 * PATCH_LENGTH + 1
img_cols = 2 * PATCH_LENGTH + 1
img_channels = data_hsi.shape[2]
INPUT_DIMENSION = data_hsi.shape[2]

data = preprocessing.scale(data)
data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
whole_data = data_
padded_data = np.lib.pad(
    whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)), 'constant',constant_values=0)

#Load model
model = models.S3KAIResNet(BAND, CLASSES_NUM, 2,PARAM_KERNEL_SIZE).cuda()
summary(model, input_data=(1, img_rows, img_cols, BAND), verbose=1)

train_indices, test_indices = utils.sampling(PARAM_VAL,gt)

net = models.S3KAIResNet(BAND, CLASSES_NUM, 2,PARAM_KERNEL_SIZE)

if PARAM_OPTIM == 'diffgrad':
    optimizer = optim2.DiffGrad(
        net.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0)  # weight_decay=0.0001)
if PARAM_OPTIM == 'adam':
    optimizer = optim.Adam(
        net.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0)

time_1 = int(time.time())

_, total_indices = utils.sampling(1, gt)

TRAIN_SIZE = len(train_indices)
print('Train size: ', TRAIN_SIZE)
TEST_SIZE =  len(test_indices)
print('Test size: ', TEST_SIZE)
VAL_SIZE = int(TRAIN_SIZE)
print('Validation size: ', VAL_SIZE)

gt_all = gt[total_indices] - 1
y_train = gt[train_indices] - 1
y_test = gt[test_indices] - 1

train_data = utils.select_small_cubic(TRAIN_SIZE, train_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION)
print("train shape : ",train_data.shape)

x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)


test_data = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION))
test_data_assign = utils.index_assignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)

for i in range(len(test_data_assign)):
        test_data[i] = utils.select_patch(padded_data, test_data_assign[i][0], test_data_assign[i][1], PATCH_LENGTH)

x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)
print("Test shape : ", x_test_all.shape)

x_val = x_test_all[-VAL_SIZE:]
y_val = y_test[-VAL_SIZE:]

x_test = x_test_all[:-VAL_SIZE]
y_test = y_test[:-VAL_SIZE]

print("X test Shape",x_test.shape)
print("Y test Shape",y_test.shape)

x1_tensor_valida = torch.from_numpy(x_val).type(torch.FloatTensor).unsqueeze(1)
y1_tensor_valida = torch.from_numpy(y_val).type(torch.FloatTensor)
torch_dataset_valida = Data.TensorDataset(x1_tensor_valida, y1_tensor_valida)

x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(1)
y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
torch_dataset_test = Data.TensorDataset(x1_tensor_test,y1_tensor_test)

train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        num_workers=0, )
valiada_iter = Data.DataLoader(
    dataset=torch_dataset_valida,  # torch TensorDataset format
    batch_size=batch_size,  # mini batch size
    shuffle=True,
    num_workers=0,
)
test_iter = Data.DataLoader(
    dataset=torch_dataset_test,  # torch TensorDataset format
    batch_size=batch_size,  # mini batch size
    shuffle=False,
    num_workers=0,
)

tic1 = time.time()

train.train(
    net,
    train_iter,
    valiada_iter,
    loss,
    optimizer,
    device,
    epochs=PARAM_EPOCH)

toc1 = time.time()

pred_test = []
tic2 = time.time()
with torch.no_grad():
    for X, y in test_iter:
        # print('Shape of X', X.shape, 'Shape of y', y.shape)
        # X = X.permute(0, 3, 1, 2)
        X = X.to(device)
        net.eval()
        y_hat = net(X)
        pred_test.extend(np.array(net(X).cpu().argmax(axis=1)))
toc2 = time.time()
collections.Counter(pred_test)
gt_test = gt[test_indices] - 1

overall_acc = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
each_acc, average_acc = record.aa_and_each_accuracy(confusion_matrix)
kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])

KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))

KAPPA.append(kappa)
OA.append(overall_acc)
AA.append(average_acc)
TRAINING_TIME.append(toc1 - tic1)
TESTING_TIME.append(toc2 - tic2)

print("--------" + " Training Finished-----------")
print('OA    : ',OA)
print('AA    : ',AA)
print('KAPPA : ',KAPPA)
print('Training Time : ', TRAINING_TIME)
print('Testing Time  : ', TESTING_TIME)