import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader,TensorDataset, random_split
import cv2
import os

class UTKFaceDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = []
        self.genders = []

        for img_file in os.listdir(directory):
            try:
                parts = img_file.split("_")
                if len(parts) < 2:  
                    print(f"Error file name: {img_file}")
                    continue  
                gender = int(parts[1])  
                img_path = os.path.join(directory, img_file)
                self.images.append(img_path)
                self.genders.append(gender)
            except ValueError as e:  
                print(f"Error with image {img_file}: {e}")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        gender = self.genders[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.transform:
            image = self.transform(image)
        return image, gender

class ConvNet(torch.nn.Module):
    def __init__(self, hidden=64, output=1):
        super(ConvNet, self).__init__()        
        self.conv1 = torch.nn.Conv2d(1, 2, kernel_size=6, padding=0, stride=9)
        #self.fc1 = torch.nn.Linear(1024, 256)
        self.fc2 = torch.nn.Linear(242, hidden)
        self.fc3 = torch.nn.Linear(hidden, output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        # the model uses the square activation function
        x = x * x
        # flattening while keeping the batch axis
        num_features = x.size(1) * x.size(2) * x.size(3)
        x = x.view(-1, num_features)
        #x = x.view(-1, 256)
        x = self.fc2(x)
        x = x * x
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
def train(model, train_loader, criterion, optimizer, n_epochs=100):

    # model in training mode
    model.train()
    for epoch in range(1, n_epochs+1):

        train_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            output = output.squeeze()  # Loại bỏ chiều có kích thước 1
            loss = criterion(output, target.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            predicted = output >= 0.5  # Ngưỡng xác suất 0.5 để phân loại
            total += target.size(0)
            correct += (predicted == target.float()).sum().item()


        # calculate average losses
        train_loss = train_loss / len(train_loader)
        accuracy = 100 * correct / total

        print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.2f}%'.format(
            epoch, train_loss, accuracy))

    return model

if __name__ == '__main__':

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

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = ConvNet()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = train(model, train_loader, criterion, optimizer, 100)
    torch.save(model, 'model_100.pth')