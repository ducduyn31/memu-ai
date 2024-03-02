import tenseal as ts 
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader,TensorDataset, random_split
import os
import numpy as np
from time import time

from gen_detection import UTKFaceDataset, ConvNet


class EncConvNet:
    def __init__(self, torch_nn):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        
        self.fc1_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc2.bias.data.tolist()
        
        self.fc2_weight = torch_nn.fc3.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc3.bias.data.tolist()

    def forward(self, enc_x, windows_nb):
        # conv layer
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
        enc_x.square_()
        # fc1 layer
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        # square activation
        enc_x.square_()
        # fc2 layer
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
def test(model, test_loader):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))

    # model in evaluation mode
    model.eval()

    for data, target in test_loader:
        output = model(data)
        # convert output probabilities to predicted class
        predicted = output >= 0.5  # Ngưỡng xác suất 0.5 để phân loại
        correct = predicted.float() == target.view_as(predicted)
        for i in range(len(target)):
            label = int(target[i].item())
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    print("Evaluate plain test set")    
    for i in range(2):
        if class_total[i] > 0:
            print(f"Accuracy of class {i} : {100 * class_correct[i] / class_total[i]:.2f}%")
        else:
            print(f"Class {i} has no samples")
    
    overall_accuracy = sum(class_correct) / sum(class_total)
    print(f"Overall Accuracy: {100 * overall_accuracy:.2f}%")


def enc_test(context, model, test_loader, kernel_shape, stride):
    t_start = time()
    # initialize lists to monitor test loss and accuracy
    #test_loss = 0.0
    correct = 0
    #total = 0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))

    for data, target in test_loader:
        # Encoding and encryption
        x_enc, windows_nb = ts.im2col_encoding(
            context, data.view(100, 100).tolist(), kernel_shape[0],
            kernel_shape[1], stride
        )
        # Encrypted evaluation
        enc_output = model(x_enc, windows_nb)
        # Decryption of result
        output = enc_output.decrypt()
        output = torch.tensor(output)
        output = torch.sigmoid(output)
        predicted = output >= 0.5 
        correct = predicted.float() == target.view_as(predicted)
        for i in range(len(target)):
            label = int(target[i].item())
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    t_end = time()
    print(f"Evaluated encrypted test_set in {int(t_end - t_start)} seconds")
    for i in range(2):
        if class_total[i] > 0:
            print(f"Accuracy of class {i} : {100 * class_correct[i] / class_total[i]:.2f}%")
        else:
            print(f"Class {i} has no samples")
    
    overall_accuracy = sum(class_correct) / sum(class_total)
    print(f"Overall Accuracy: {100 * overall_accuracy:.2f}%")

if __name__ == '__main__':
    model = torch.load('./model_100.pth')
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])

    dir = os.path.dirname(os.path.realpath(__file__)) + '/utk_10000/'

    dataset = UTKFaceDataset(dir, transform=transform)

    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    #train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    kernel_shape = model.conv1.kernel_size
    stride = model.conv1.stride[0]

    ## Encryption Parameters
    # controls precision of the fractional part
    bits_scale = 26

    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
    )

    # set the scale
    context.global_scale = pow(2, bits_scale)

    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()

    enc_model = EncConvNet(model)
    test(model,test_loader)
    enc_test(context, enc_model, test_loader, kernel_shape, stride)
