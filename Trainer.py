import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward
import numpy as np
from math import ceil
import argparse
from CONSTANTS import BATCH_SIZE, PATCH_SIZE, IMG_SIZE, FRAME_NUM
import zipfile
from tqdm import tqdm
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from vivit import Transformer, ViViT


class VIVIT_Trainer():
    def __init__(self, model, optimizer, criterion):
        # CONFIGURE TORCH DEVICE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'using {device} device...')

        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
    
    def accuracy(self, y_pred:np.ndarray, target:np.ndarray):
        correct = int(np.sum(y_pred == target))
        num = y_pred.shape[0]
        acc = correct/num
        return acc



    def train(self, X_train, X_test, y_train, y_test, BATCH_SIZE:int=4, num_epoch:int=10):
        # Initialization
        BATCH = ceil(X_train.shape[0] / BATCH_SIZE)
        best_acc = 0
        sigmoid= nn.Sigmoid()

        # Use cpu for validation by hardware limitations
        X_test = X_test.type(torch.float32).to(torch.device('cpu'))
        y_test = y_test.type(torch.float32).to(torch.device('cpu'))

        for epoch in tqdm(range(num_epoch)):
            train_loss = 0
            # Training
            model_test = self.model.to(torch.device(self.device))
            self.model.train()
            self.optimizer.zero_grad()
            batch_count = 0
            while batch_count < BATCH:
                X_train_batch = X_train[batch_count*BATCH_SIZE:] if batch_count == BATCH-1 else X_train[batch_count*BATCH_SIZE:(batch_count+1)*BATCH_SIZE]
                y_train_batch = y_train[batch_count*BATCH_SIZE:] if batch_count == BATCH-1 else y_train[batch_count*BATCH_SIZE:(batch_count+1)*BATCH_SIZE]
                X_train_batch = X_train_batch.type(torch.float32).to(self.device)
                y_train_batch = y_train_batch.type(torch.float).to(self.device)

                outputs = self.model(X_train_batch)

                outputs = sigmoid(outputs)
                # print((outputs, y_train_batch))
                loss = self.criterion(outputs, y_train_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                batch_count += 1
            train_loss /= BATCH

            # print(f'Epoch {epoch + 1}, Training Loss: {loss.item()}, Test Loss: {test_loss.item()}, Accuracy: {accuracy}')
            # print(f'Epoch {epoch + 1}, Training Loss: {loss.item()}')


            # Validation
            self.model.to(torch.device('cpu'))
            self.model.eval() 
            test_outputs = self.model(X_test)
            # print(test_outputs)
            
            test_outputs = sigmoid(test_outputs)
            test_loss = self.criterion(test_outputs, y_test)

            test_outputs[test_outputs>=0.5] = 1
            test_outputs[test_outputs<0.5] = 0
            y_pred = test_outputs.detach().numpy()
            target = y_test.detach().numpy()
            # print('pred:',y_pred)
            # print('truth: ',target)
            val_acc = self.accuracy(y_pred, target)

            print(f'Epoch {epoch + 1}, Training Loss: {train_loss}, Test Loss: {test_loss.item()}, Accuracy: {val_acc}')

            # TODO: model.save() on best epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                }, 'fully_trained_model.pth')

                print("Better validation accuracy, model saved successfully!")

    










# if __name__ == "__main__":
#     # CONFIGURE TORCH DEVICE
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f'using {device} device...')


#     # LOAD DATASET
#     img_dataset = []
#     img_archive = zipfile.ZipFile('Processing_results/filtered_image_stack.zip', 'r')

#     for file in tqdm(img_archive.infolist()):
#         if file.filename.endswith('.npy'):
#             with img_archive.open(file) as npy_file:
#                 img_dataset.append(np.load(io.BytesIO(npy_file.read())))
#     print(len(img_dataset))

#     # find max. t
#     t_max = 0
#     w = 480
#     h = 640
#     c = 3
#     for img in img_dataset:
#         t_max = max(t_max, img.shape[0])
#     # ensure unified frames
#     assert img.shape[1] == w and img.shape[2] == h and img.shape[3] == c
#     print(t_max)

#     # padding & reshape
#     img_tensors = []
#     for img in tqdm(img_dataset):
#         img_tensors.append(torch.tensor(img))
#     train_dataloader = torch.nn.utils.rnn.pad_sequence(img_tensors, batch_first=True)
#     print('N,T,H,W,C',train_dataloader.shape) # (sample_size,frames,height,width,channel)
#     N,T,H,W,C = train_dataloader.shape
#     train_dataloader = train_dataloader.reshape(N,T,C,H,W)
#     print('N,T,C,H,W',train_dataloader.shape)

#     # Labels
#     label_archive = zipfile.ZipFile('MU3D_dataset.zip', 'r')
#     for file in tqdm(label_archive.infolist()):
#         if file.filename.endswith('.xlsx'):
#             with label_archive.open(file) as label_file:
#                 label = pd.read_excel(label_file, sheet_name='Video-Level Data')

#     y = torch.tensor(label['Veracity'].astype('float').values)
#     y = y[:61] #TODO: temporaty, sould be full sample_size
#     y = y.unsqueeze(1)

#     # train / test split 
#     X_train, X_test, y_train, y_test = train_test_split(train_dataloader, y, test_size=0.1, random_state=42)

#     # Test PyTorch sensors
#     X_test = X_test.type(torch.float32).to(device)
#     y_test = y_test.type(torch.long).to(device).squeeze()
#     print('train data szie:',X_train.shape, y_train.shape)
#     print('test data size:',X_test.shape, y_test.shape)

#     FRAME_NUM = T #TODO: temporaty, sould be full sample_size
#     model = ViViT(image_size_w=W, image_size_h=H,
#                    patch_size_w=80, patch_size_h=60,
#                      num_classes=2,
#                      num_frames=FRAME_NUM).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()

    

    
#     # PARSE ARGUMENTS
#     FUNCTION_MAP = {'train' : train(model, EPOCH, X_train, X_test, y_train, y_test, optimizer, criterion)
# }
#     """
#         run "python vivit --train" for vivit training
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument('command', choices=FUNCTION_MAP.keys(),
#                         help="{train:trains vivit model}")
#     args = parser.parse_args()

#     func = FUNCTION_MAP[args.command]
#     func()
