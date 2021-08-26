import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super().__init__()
        # Embedded layer that transforms the output of the CNN to the input format of the RNN
        self.word_embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        
        
        # Discard the <end> word to avoid error.
        captions = captions[:, :-1]
        captions = self.word_embedding_layer(captions)
        
        # Concatenate the feature vctors for image and captions
        # (batch_size, embed_size) + (batch_size, caption_lenght, embed_size)
        inputs_lstm = torch.cat((features.unsqueeze(1), captions), dim=1)
        
        outputs_lstm, _ = self.lstm(inputs_lstm)
        
        # output shape : (batch_size, caption length, vocab_size)
        output_net = self.fc(outputs_lstm)
        
        return output_net

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        outputs_net = []
        output_len = 0
        
        while(output_len != max_len + 1):
            
            out, states = self.lstm(inputs, states)
            
            out = self.fc(out.squeeze(dim=1))
            _, predicted_index = torch.max(out, 1)
            
            # CUDA Tensor -> cpu -> numpy
            outputs_net.append(predicted_index.cpu().numpy()[0].item())
        
            # If <end> character -> break
            if(predicted_index == 1):
                break
                
            # Prepare inputs for the next loop iteration.
            inputs = self.word_embedding_layer(predicted_index)
            inputs = inputs.unsqueeze(1)
            
            output_len += 1
            
        return outputs_net