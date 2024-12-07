import torch.nn as nn
import torch
from tqdm import tqdm
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self,in_channels, out_channels):
        """
        output same as input
        """
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(inplace=False))  # Changed inplace to False
        self.conv2 = nn.Sequential(
                        nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(inplace=False))
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        if(in_channels != out_channels):
            self.residual = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.in_channels != self.out_channels:
            residual = self.residual(x)
        else:
            residual = x
        return F.relu(out + residual, inplace=False)


class ResNetLike(nn.Module):
    def __init__(self, input = 201, input_ch = 1, num_classes = 2):
        super(ResNetLike, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_ch, 64, kernel_size=7, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResNetBlock(64,64),
            ResNetBlock(64,64),
            ResNetBlock(64,64),
            ResNetBlock(64,128),    # out 1 x 128 x n
            nn.MaxPool1d(2),        # out 1 x 128 x n//2
            ResNetBlock(128,128),
            ResNetBlock(128,128),
            ResNetBlock(128,256),
            nn.MaxPool1d(2),        # out 1 x 256 x n//2
            ResNetBlock(256,256),
            ResNetBlock(256,256),
            ResNetBlock(256,512),
            nn.MaxPool1d(2),        # out 1 x 512 x n//8
            nn.Flatten(),
            nn.Linear(512*(input//8), 256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.model.to('cuda:0')

    def forward(self, x):

        return self.model(x)
    
    def train_model(self, train_loader, valid_loader, num_epochs = 5, learning_rate=0.001, save_best = False, save_thr = 0.94):
        best_accuracy = 0.0
        total_step = len(train_loader)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

        for epoch in range(num_epochs):
            # self.train()
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(tqdm(train_loader)):
                # Move tensors to the configured device
                images = images.float().to("cuda")
                labels = labels.type(torch.LongTensor)
                labels = labels.to("cuda")


                optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                # Backward and optimize
                loss.backward()
                
                optimizer.step()

                # accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct += (torch.eq(predicted, labels)).sum().item()
                total += labels.size(0)

                del images, labels, outputs

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'
                            .format(epoch+1, num_epochs, i+1, total_step, loss.item(), (float(correct))/total))


            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Validation
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in valid_loader:
                    images = images.float().to("cuda")
                    labels = labels.to("cuda")
                    outputs = self.forward(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (torch.eq(predicted, labels)).sum().item()
                    del images, labels, outputs
                if(((100 * correct / total) > best_accuracy) and save_best and ((100 * correct / total) > save_thr)):
                    torch.save(self.state_dict(), "best_resnet50_MINST-DVS2.pt")

                print('Accuracy of the network: {} %'.format( 100 * correct / total))