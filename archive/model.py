import os
from tqdm import tqdm
import torch
import numpy as np
import math

##################################################################
######################## Channel Attention #######################
##################################################################
class SelfAttentionChannel(torch.nn.Module):
    def __init__(self,  dim_model,
                        num_heads,  
                        dropout_rate):
        super(SelfAttentionChannel, self).__init__()
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.wq = torch.nn.Linear(in_features = self.dim_model, out_features =self.dim_model, bias = True)
        self.wk = torch.nn.Linear(in_features =self.dim_model, out_features =self.dim_model, bias = False)
        self.wv = torch.nn.Linear(in_features =self.dim_model, out_features =self.dim_model, bias = False)
        self.linear = torch.nn.Linear(dim_model, dim_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
    
    def _split_heads(self, x, batch_size):
        # before : (batch_size, sequence_length, dim_model)
        # split dim_model -> num_heads, depth and transpose 
        # after  : (batch_size, num_heads, sequence_length, depth)
        # print(x.size(-1))
        # print(self.num_heads)
        x = x.reshape(batch_size, -1, self.num_heads, x.size(-1) // self.num_heads)
        return x.transpose(1, 2)

    def _calculate_attention(self, q, k, v):
        # print("q :", q.shape)
        # print("k :", k.shape)
        # print("v :", v.shape)
        att_score = torch.matmul(q, k.transpose(2,3))
        dk = q.shape[-1]
        scaled_att_score = att_score / math.sqrt(dk)
        attention_prob = torch.nn.functional.softmax(scaled_att_score)
        attention_prob = self.dropout(attention_prob)
        final_score = torch.matmul(attention_prob, v)
        return final_score

    def forward(self, q, k, v):
        batch_size = q.size(0)
        # (batch size, sequence_length, dim_model)
        q = self.wq(q) 
        k = self.wk(k)
        v = self.wk(v)

        #split the unihead into multihead
        # (batch size, num_heads, sequence_length, depth)
        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        
        attention_score = self._calculate_attention(q, k, v) # (batch_size, num_heads, seq_length, depth)
        attention_score = attention_score.transpose(1, 2) # (batch_size, seq_length, num_heads, depth)
        attention_score = attention_score.contiguous().view(batch_size, -1, self.dim_model) # (batch_size, seq_length, dim_model)
        attention_score = self.linear(attention_score) # (n_batch, seq_len, d_embed)
        return attention_score   

class FeedForwardChannel(torch.nn.Module):
    def __init__(self, dim_model, dim_inner):
        super(FeedForwardChannel, self).__init__()
        self.linear1 = torch.nn.Linear(dim_model, dim_inner)
        self.linear2 = torch.nn.Linear(dim_inner, dim_model)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
        
class SqueezeExcitationChannel(torch.nn.Module):
    def __init__(self, dim_model,
                        sec_in_channels,
                        squeeze_factor,
                        sec_kernel_size,
                        sec_stride):
        super(SqueezeExcitationChannel, self).__init__()
        C = int(np.floor(sec_in_channels / squeeze_factor))

        #squeeze
        self.conv1 = torch.nn.Conv1d(in_channels = sec_in_channels, 
                                      out_channels = C,
                                      kernel_size = sec_kernel_size,
                                     stride = sec_stride,
                                     padding = 1)
        self.batchnorm1 = torch.nn.BatchNorm1d(C)
        #excitation
        self.conv2 = torch.nn.Conv1d(in_channels = C,
                                     out_channels = sec_in_channels,
                                     kernel_size = sec_kernel_size,
                                     stride = sec_stride,
                                     padding = 1)
        self.batchnorm2 = torch.nn.BatchNorm1d(sec_in_channels)
        self.act = torch.nn.LeakyReLU()      
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.act(x)
        return x

class ChannelBCEncoder(torch.nn.Module):
    def __init__(self, 
                dim_model,
                num_channels,
                num_samples,
                num_heads,
                dropout_rate,
                dim_inner,
                squeeze_factor,
                sec_in_channels,
                sec_kernel_size,
                sec_stride):
        super(ChannelBCEncoder, self).__init__()
        # (c) BC encoder
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, dim_model))
        # self.layernorm_MSA = torch.nn.LayerNorm((num_samples, num_channels, dim_model))
        # self.layernorm_FFN = torch.nn.LayerNorm((num_samples, num_channels, dim_model))
        self.layernorm_MSA = torch.nn.LayerNorm(( num_channels, dim_model))
        self.layernorm_FFN = torch.nn.LayerNorm((num_channels, dim_model))
        
        self.MSA = SelfAttentionChannel(dim_model = dim_model,
                                        num_heads = num_heads,
                                        dropout_rate = dropout_rate)
        self.FFN = FeedForwardChannel(dim_model = dim_model,
                                        dim_inner = dim_inner)
        self.SEC = SqueezeExcitationChannel(dim_model = dim_model,
                                            sec_in_channels = sec_in_channels,
                                            squeeze_factor = squeeze_factor,
                                            sec_kernel_size = sec_kernel_size,
                                            sec_stride = sec_stride)
    def forward(self, x):
        # (c) BC encoder
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)    
        out = self.MSA(x, x, x)
        out = self.layernorm_MSA(out) + x
        out2 = self.FFN(x)
        out2 = self.layernorm_FFN(out2) + out
        out3 = self.SEC(out2) + out2
        return out3

class ChannelAttention(torch.nn.Module):
    def __init__(self, first_in_channels,
                        first_out_channels,
                        first_kernel_size,
                        second_in_channels,
                        second_out_channels,
                        second_kernel_size,
                        dim_model,
                        dim_inner,
                        num_samples,
                        length,
                        num_heads,
                        dropout_rate,
                        squeeze_factor,
                        sec_kernel_size,
                        sec_stride
                        ):
        super(ChannelAttention, self).__init__()
        # (a) two convolution layers
        self.conv_1 = torch.nn.Conv1d(in_channels = first_in_channels, 
                                      out_channels = first_out_channels,
                                      kernel_size = first_kernel_size)
        self.norm_1 = torch.nn.BatchNorm1d(num_features = first_out_channels)
        self.act = torch.nn.LeakyReLU()
        self.conv_2 = torch.nn.Conv1d(in_channels = second_in_channels,
                                      out_channels = second_out_channels,
                                      kernel_size = second_kernel_size)
        self.norm_2 = torch.nn.BatchNorm1d(num_features = second_out_channels)
        # (b) linear token embedding
        self.linear_embedding1 = torch.nn.Linear(in_features = length - 6, 
                                                 out_features = dim_model)
        self.BCEncoder = ChannelBCEncoder(dim_model = dim_model,
                                        num_channels = second_out_channels + 1,
                                        num_samples = num_samples,
                                        num_heads = num_heads,
                                        dropout_rate = dropout_rate,
                                        dim_inner = dim_inner,
                                        squeeze_factor = squeeze_factor,
                                        sec_in_channels = second_out_channels + 1,
                                        sec_kernel_size = sec_kernel_size,
                                        sec_stride = sec_stride
                                        )

    def forward(self, x):
        # (a) two convolution layers
        # print(x.shape)
        # print(x.type())
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.act(x)
        x = self.conv_2(x)
        x = self.norm_2(x)
        x = self.act(x)
        # (b) linear token embedding
        x = self.linear_embedding1(x)
        x = self.BCEncoder(x)
        x_rep = x[:,0]
        # print(x_rep.shape)
        return x_rep

##################################################################
######################### Slice Attention ########################
##################################################################
class SelfAttentionSlice(torch.nn.Module):
    def __init__(self,  dim_model,
                        num_heads, 
                        dropout_rate):
        super(SelfAttentionSlice, self).__init__()
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.wq = torch.nn.Linear(in_features = self.dim_model, out_features =self.dim_model, bias = True)
        self.wk = torch.nn.Linear(in_features =self.dim_model, out_features =self.dim_model, bias = False)
        self.wv = torch.nn.Linear(in_features =self.dim_model, out_features =self.dim_model, bias = False)
        self.linear = torch.nn.Linear(dim_model, dim_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
    
    def _split_heads(self, x, batch_size):
        # before : (batch_size, sequence_length, dim_model)
        # split dim_model -> num_heads, depth and transpose 
        # after  : (batch_size, num_heads, sequence_length, depth)
        # print(x.size(-1))
        # print(self.num_heads)
        x = x.reshape(batch_size, -1, self.num_heads, x.size(-1) // self.num_heads)
        return x.transpose(1, 2)

    def _calculate_attention(self, q, k, v):
        # print("q :", q.shape)
        # print("k :", k.shape)
        # print("v :", v.shape)
        att_score = torch.matmul(q, k.transpose(2,3))
        dk = q.shape[-1]
        scaled_att_score = att_score / math.sqrt(dk)
        attention_prob = torch.nn.functional.softmax(scaled_att_score)
        attention_prob = self.dropout(attention_prob)
        final_score = torch.matmul(attention_prob, v)
        return final_score

    def forward(self, q, k, v):
        batch_size = q.size(0)
        # (batch size, sequence_length, dim_model)
        q = self.wq(q) 
        k = self.wk(k)
        v = self.wk(v)

        #split the unihead into multihead
        # (batch size, num_heads, sequence_length, depth)
        # print(q.shape)
        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        
        attention_score = self._calculate_attention(q, k, v) # (batch_size, num_heads, seq_length, depth)
        attention_score = attention_score.transpose(1, 2) # (batch_size, seq_length, num_heads, depth)
        attention_score = attention_score.contiguous().view(batch_size, -1, self.dim_model) # (batch_size, seq_length, dim_model)
        attention_score = self.linear(attention_score) # (n_batch, seq_len, d_embed)
        return attention_score   

class FeedForwardSlice(torch.nn.Module):
    def __init__(self, dim_model, dim_inner):
        super(FeedForwardSlice, self).__init__()
        self.linear1 = torch.nn.Linear(dim_model, dim_inner)
        self.linear2 = torch.nn.Linear(dim_inner, dim_model)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
        
class SqueezeExcitationSlice(torch.nn.Module):
    def __init__(self, dim_model,
                        sec_first_in_channels,
                        squeeze_factor,
                        sec_kernel_size,
                        sec_stride):
        super(SqueezeExcitationSlice, self).__init__()
        C = int(np.floor(sec_first_in_channels / squeeze_factor))

        #squeeze
        self.conv1 = torch.nn.Conv1d(in_channels = sec_first_in_channels, 
                                      out_channels = C,
                                      kernel_size = sec_kernel_size,
                                     stride = sec_stride,
                                     padding = 1)
        self.batchnorm1 = torch.nn.BatchNorm1d(C)
        #excitation
        self.conv2 = torch.nn.Conv1d(in_channels = C,
                                     out_channels = sec_first_in_channels,
                                     kernel_size = sec_kernel_size,
                                     stride = sec_stride,
                                     padding = 1)
        self.batchnorm2 = torch.nn.BatchNorm1d(sec_first_in_channels)
        self.act = torch.nn.LeakyReLU()      
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.act(x)
        return x

class SliceBCEncoder(torch.nn.Module):
    def __init__(self, dim_model,
                 num_samples,
                 num_channels,
                 dim_inner,
                 num_heads,
                 dropout_rate,
                 sec_first_in_channels,
                 squeeze_factor,
                 sec_kernel_size,
                 sec_stride

                 ):
        super(SliceBCEncoder, self).__init__()
        # (c) BC encoder
        
        self.layernorm_MSA = torch.nn.LayerNorm((num_samples, num_channels, dim_model))
        self.layernorm_FFN = torch.nn.LayerNorm((num_samples, num_channels, dim_model))
        self.MSA = SelfAttentionSlice(dim_model = dim_model,
                                      num_heads = num_heads, 
                                      dropout_rate = dropout_rate)
        self.FFN = FeedForwardSlice(dim_model = dim_model,
                                    dim_inner = dim_inner)
        self.SEC = SqueezeExcitationSlice(dim_model = dim_model,
                                        sec_first_in_channels = sec_first_in_channels,
                                        squeeze_factor = squeeze_factor,
                                        sec_kernel_size = sec_kernel_size,
                                        sec_stride = sec_stride)

    def forward(self, x):
        
        # (c) BC encoder
        
        out = self.MSA(x, x, x)
        out = self.layernorm_MSA(out) + x
        out2 = self.FFN(x)
        out2 = self.layernorm_FFN(out2) + out
        out3 = self.SEC(out2) + out2
        return out3

class SliceAttention(torch.nn.Module):
    def __init__(self, 
                 N ,
                 L,
                 first_in_channels,
                 first_out_channels,
                 first_kernel_size,
                 second_in_channels,
                 second_out_channels,
                 second_kernel_size,
                 dim_model,
                 dim_inner,
                num_samples,
                # num_channels,
                length,
                num_heads,
                dropout_rate ,
                squeeze_factor ,
                sec_kernel_size,
                sec_stride
                 ):
        super(SliceAttention, self).__init__()
        self.N = N
        self.dim_model = dim_model
        
        self.length = int(np.floor(L/N))
        self.slice1conv = torch.nn.Conv1d(in_channels = first_in_channels,
                                          out_channels = first_out_channels,
                                          kernel_size = first_kernel_size)
        self.slice2conv = torch.nn.Conv1d(in_channels = first_in_channels,
                                          out_channels = first_out_channels,
                                          kernel_size = first_kernel_size)
        self.slice3conv = torch.nn.Conv1d(in_channels = first_in_channels,
                                          out_channels = first_out_channels,
                                          kernel_size = first_kernel_size)
        self.sliceconv = torch.nn.Conv1d(in_channels = second_in_channels,
                                          out_channels = second_out_channels,
                                          kernel_size = second_kernel_size)
        self.slice1norm = torch.nn.BatchNorm1d(num_features = first_out_channels)
        self.slice2norm = torch.nn.BatchNorm1d(num_features = first_out_channels)
        self.slice3norm = torch.nn.BatchNorm1d(num_features = first_out_channels)
        self.slicenorm = torch.nn.BatchNorm1d(num_features = second_out_channels)
        self.act = torch.nn.LeakyReLU()
        
        self.ChannelFusion = torch.nn.Conv1d(in_channels = second_out_channels,
                                             out_channels = 1,
                                             kernel_size = second_kernel_size,
                                             padding = 1)
        # (b) linear token embedding
        self.linear_embedding = torch.nn.Linear(in_features = self.length - 6, 
                                                 out_features = dim_model)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, dim_model))
        
        self.BCEncoder = SliceBCEncoder(dim_model=dim_model,
                                         num_samples = num_samples,
                                         num_channels = N + 1,
                                         dim_inner = dim_inner,
                                         num_heads = num_heads,
                                         dropout_rate = dropout_rate,
                                         sec_first_in_channels = N + 1,
                                         squeeze_factor = squeeze_factor,
                                         sec_kernel_size = sec_kernel_size,
                                         sec_stride = sec_stride
                                        )

    def _position_embedding(self, X):
        def _even_position(pos, i, dim_model):
            return math.sin(pos / (10000**(i / dim_model)))
        def _odd_position(pos, i, dim_model):
            return math.cos(pos / (10000**((i-1) / dim_model)))

        pe = torch.zeros((X.size(1), self.dim_model))
        for pos in range(X.size(1)):
            for i in range(self.dim_model):
                if i % 2 == 0: #even
                    pe[pos, i] = _even_position(pos, i, self.dim_model)
                else:
                    pe[pos, i] = _odd_position(pos, i, self.dim_model)
        return pe
    


    def forward(self, x):
        # channels = torch.Tensor([])
        slice_length = int(np.floor(x.size(-1)/self.N))
        i = 0
        
        x1 = x[:, :, i:i+slice_length]
        # print(x1.shape)
        x2 = x[:, :, i+slice_length:i+slice_length*2]
        x3 = x[:, :, i+slice_length*2:i+slice_length*3]
        # X1
        x1 = self.slice1conv(x1)
        x1 = self.slice1norm(x1)
        x1 = self.act(x1)
        # X2
        x2 = self.slice2conv(x2)
        x2 = self.slice2norm(x2)
        x2 = self.act(x2)
        # X3
        x3 = self.slice3conv(x3)
        x3 = self.slice3norm(x3)
        x3 = self.act(x3)

        # shared
        # print(x1.grad)
        x1 = self.sliceconv(x1)
        x1 = self.slicenorm(x1)
        x1 = self.act(x1)
        x2 = self.sliceconv(x2)
        x2 = self.slicenorm(x2)
        x2 = self.act(x2)
        x3 = self.sliceconv(x3)
        x3 = self.slicenorm(x3)
        x3 = self.act(x3)
        # print(x1.shape, x2.shape, x3.shape) # v : num of last filter : 16, df : output dim : 27

        #ChannelFusion
        x1 = self.ChannelFusion(x1) 
        x2 = self.ChannelFusion(x2) 
        x3 = self.ChannelFusion(x3) 
        G = torch.cat((x1, x2, x3), dim=1)
        
        # Linear Token Embedding
        x = self.linear_embedding(G)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)    
        pe = self._position_embedding(x)

        x = self.BCEncoder(x)
        x_rep = x[:,0]
        # print(x_rep.shape)
        return x_rep


class Net(torch.nn.Module):
    def __init__(self,
                dim_model = 64,
                gamma = 2,
                u = 2,
                ys = None,
                first_in_channels = 6,
                first_out_channels = 32,
                first_kernel_size = 5,
                second_in_channels = 32,
                second_out_channels = 16,
                second_kernel_size = 3,
                N = 3,  #slice
                L = 100, # slice
                dim_inner = 128,
                num_samples = 40, 
                length = 100, 
                num_heads = 8,
                dropout_rate = 0.15,
                squeeze_factor = 2,
                sec_kernel_size = 3,
                sec_stride = 1,
                num_classes = 2):
        super(Net, self).__init__()
        self.channelattention = ChannelAttention(first_in_channels = first_in_channels,
                                                    first_out_channels = first_out_channels,
                                                    first_kernel_size = first_kernel_size,
                                                    second_in_channels = second_in_channels,
                                                    second_out_channels = second_out_channels,
                                                    second_kernel_size = second_kernel_size,
                                                    dim_model = dim_model,
                                                    dim_inner = dim_inner,
                                                    num_samples = num_samples, 
                                                    length = length, 
                                                    num_heads = num_heads,
                                                    dropout_rate = dropout_rate,
                                                    squeeze_factor = squeeze_factor,
                                                    sec_kernel_size = sec_kernel_size,
                                                    sec_stride = sec_stride)
        self.sliceattention = SliceAttention(N = N, 
                                             L = L, 
                                             first_in_channels = first_in_channels,
                                             first_out_channels = first_out_channels,
                                             first_kernel_size = first_kernel_size,
                                             second_in_channels = second_in_channels,
                                             second_out_channels = second_out_channels,
                                             second_kernel_size = second_kernel_size,
                                             dim_model = dim_model,
                                             dim_inner = dim_inner,
                                            num_samples = num_samples, 
                                            # num_channels = N + 1, # 

                                            length = length, 
                                            num_heads = num_heads,
                                            dropout_rate = dropout_rate,
                                            squeeze_factor = squeeze_factor,
                                            sec_kernel_size = sec_kernel_size,
                                            sec_stride = sec_stride)
        self.num_classes = num_classes
        self.d = int(dim_model / gamma)
        self.fc = torch.nn.Linear(dim_model * 2, self.d)
        self.u = u
        self.ys = ys
        self.attention_V = torch.nn.Sequential(
            torch.nn.Linear(self.d, self.u),
            torch.nn.Tanh()
        )
        self.attention_weight = torch.nn.Linear(self.u, 1)
        self.attention_softmax = torch.nn.Softmax(dim = 1)
        self.classifier = torch.nn.Softmax(dim = -1)# self.A = torch.
        self.idx = 0

    def _prototype_learning(self, X_rep, y, classes):
        C_all = []
        y = y[self.idx]
        # print(X_rep.shape)
        for cls in classes:
            H = []#torch.zeros(X_rep.shape)
            for i in range(X_rep.size(0)):
                if y[i] == cls:
                    H.append(X_rep[i, :])
            # print("H", len(H), H[0].shape)
            print(H)
            H = torch.stack(H)
            # print(cls, ":",H.shape)
            A = self.attention_V(H)
            wA = self.attention_weight(A)
            wA = torch.transpose(wA, 1, 0)
            wA = self.attention_softmax(wA)

            class_prototype = torch.mm(wA, H)
            C_all.append(class_prototype)
            # print(class_prototype.shape)
        C_all = torch.stack(C_all)#.squeeze(1)
        # print(C_all.shape)
        # print(X_rep.shape
        self.idx += 1
        return C_all

    def calculate_distance(self, x, C):
        dists = []
        for c in C:
            dist = -1 * (x - c).pow(2).sum(-1).sqrt()
            dists.append(dist)
        dists = torch.stack(dists)
        # print(dists.shape)
        probs = torch.nn.functional.softmax(dists, dim = 0)
        # probs = self.classifier(dists)
        # print(probs.shape)
        return probs
    

    def calculate_objective(self, x, y):
        # y=y.float()
        # print(x.shape)
        # x = x.double()
        y_hat = self.forward(x)
        y_hat = y_hat.float()
        y = y.float()
        numeric_y = torch.argmax(y, dim=1)
        # print(y_hat.shape)
        # print(numeric_y.long())
        y_hat = torch.clamp(y_hat, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * y_hat.log()[np.arange(y_hat.size(0)),numeric_y.long()] 

        
        # print(neg_log_likelihood.shape)
        return neg_log_likelihood

    def calculate_classification_error(self, x, y):
        y_hat = self.forward(x)
        # print(y_hat)

        _, max_values = torch.max(y_hat, dim=1)
        # print(max_values)
        max_values = max_values.long()
        one_hot_y = torch.zeros_like(y_hat)
        
        one_hot_y[torch.arange(y_hat.size(0)), max_values] = 1

        error = 1. #- y.eq(one_hot_y).float().mean()
        return error

    # def
    def forward(self, x):
        # print(self.d)
        x1 = self.channelattention(x)
        x2 = self.sliceattention(x)
        
        # print(x1.shape, x2.shape)
        x = torch.cat((x1, x2), dim = -1)
        x = self.fc(x) # 1 x 32
        # x = torch.unsqueeze(x, 1 )
        # print("X shape", x.shape)
        
        prototypes = self._prototype_learning(x, self.ys, classes=[i for i in range(self.num_classes)]) # C_all을 구함
        # x = x.unsqueeze(1)
        # print("xrep shape", x.shape) # 40, 1, 32
        # print(prototypes.shape) # 4, 1, 32
        x = self.calculate_distance(x, prototypes)
        # print(a.shape)
        x = x.transpose(0, 1) # batch_size, class, 1, dim_model

        return x