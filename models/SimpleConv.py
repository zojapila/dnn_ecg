
from torchvision.io import read_image
import torch.nn as nn
import torch
from tqdm import tqdm

class SimpleConv(nn.Module):
    def __init__(self, input = 201, input_ch = 1, num_classes = 2):
        super(SimpleConv, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_ch, 64, kernel_size=7, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding='same'),      # out 1 x 128 x n
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),                        # out 1 x 128 x n//2
            nn.Conv1d(128, 128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding='same'),     # out 1 x 256 x n//2
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),                        # out 1 x 256 x n//4
            nn.Conv1d(256, 256, kernel_size=3, padding='same'),     
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding='same'),     
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),                        # out 1 x 512 x n//8
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