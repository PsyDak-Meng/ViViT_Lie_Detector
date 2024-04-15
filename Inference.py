import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from emotion_detection.EmotionDetector import EmotionDetector
import cv2

class VIVIT_Classifier():
    def __init__(self, model) -> None:
        self.model = model


    def inference(self, X, y):
        sigmoid = nn.Sigmoid()

        while len(X.shape) < 5:
            X = X.unsqueeze(0)
            print('incorrect input shape !!')
        self.model.eval()
        self.model.to(torch.device('cpu'))

        self.X = X
        self.y = y

        outputs = self.model(X)
        outputs = sigmoid(outputs)
        confidence = float(outputs[0])*100
        outputs[outputs>=0.5] = 1
        outputs[outputs<0.5] = 0


        
        # 1 truth/ 0 lie
        print(f'The person is predicted to be truthful.') if outputs[0] == 1 else print('The person is predicted to be lying.')
        print(f'The model has {confidence:.2f}% confidence, ' + 'and the prediction is correct!' if outputs[0]==y else f'and the prediction is incorrect!')
        print(f'The predicted label is {outputs}, the true label is {y}.')

        return outputs


    #TODO: Extract attention weights
    def extract_attn_weights(self, n:int) -> None:
        # set how many frames to retrieve
        self.n = n 
        print("\nModel attention layers:", self.model.temporal_transformer.layers[0][0].fn)

        # activation = {}
        # def get_activation(name):
        #     def hook(model, input, output):
        #         activation[name] = output.detach()
        #     return hook
        # register the forward hook
        # print(model.temporal_transformer.layers[0][0].fn.attn_weights.shape)
        # model.temporal_transformer.layers[0][0].fn.attn_weights.register_forward_hook(get_activation('temporal_attn_weights'))
        # attn_weights = activation['spatial_transformer_qkv']


        # this is what you're looking for
        attn_weights = self.model.temporal_transformer.layers[0][0].fn.attn_weights
        # leave out the first cls token
        attn_weights = attn_weights[:,1:,:]
        # print('attention weights shape: ',attn_weights.shape)
        attn_weights = torch.mean(attn_weights,dim=2)
        attn_weights = nn.functional.softmax(attn_weights, dim=1).squeeze().detach().numpy()
        print('attention weights shape: ',attn_weights.shape)

        # obtain top k attn_weights that are not padded
        top_n_idx = []
        attn_idx = attn_weights.argsort()[:][::-1]
        for idx in attn_idx:
            if len(top_n_idx) <n and torch.sum(self.X.squeeze(0)[idx,:,:,:]):
                top_n_idx.append(idx)
            
        print(f'Top {self.n} significant attention frames:',top_n_idx, \
              '\ncorresponding weights: ', attn_weights[top_n_idx])
        # print(attn_weights)

        images = []
        for idx in top_n_idx:
            images.append(self.X.squeeze(0)[idx,:,:,:].squeeze(0))

        # TODO: add sentiment analysis

        # TODO: display frame
        def display_images(images: list):
            """
            Display n images from a NumPy array.
            
            Parameters:
                images (np.ndarray): Array of images with shape (num_images, height, width, channels).
                n (int): Number of images to display. Default is 4.
            """
            # Initialization
            row = self.n//2
            col = 2
            fig, axes = plt.subplots(row, col, figsize=(12, 16))  # Adjust figsize as needed
            emotion_detector = EmotionDetector('emotion_detection/model.h5')
            
            count = 0
            for i in range(row):
                for j in range(col):          
                    image = images[count].permute(1, 2, 0).detach().numpy()
                    # image = image.astype(int)
                    image = np.uint8(image)
                    

                    emotions, confidence = emotion_detector.predict(image)
                    confidence = sum(confidence)/len(confidence)
                    emotions = ','.join(emotions)

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    axes[i,j].imshow(image)
                    axes[i,j].axis('off')  # Hide axis
                    
                    frame_num = top_n_idx[count]
                    axes[i,j].set_title(f"Frame {frame_num}: {emotions} ({confidence:.2f}%)")
                    count += 1
            # print(image.shape)
            # print(image)
            
            plt.show()
        
        display_images(images)


