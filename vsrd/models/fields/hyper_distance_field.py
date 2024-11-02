import operator

import torch
import torch.nn as nn


class HyperDistanceField(nn.Module):
    '''
    该类是一个 PyTorch 神经网络模块，
    用于生成超网络 Hypernetwork 并使用它来计算距离场 Distance Field 。超网络通过给定的嵌入 Embeddings 生成主网络 Primary Network 的权重。
    
    '''

    def __init__(
        self,
        in_channels,
        out_channels_list,
        hyper_in_channels,
        hyper_out_channels_list,
    ):
        super().__init__()
        
        '''Define the MLP Architecture'''

        # input is 48, output is a 4 layer: 16,16,16
        in_channels_list = [in_channels, *out_channels_list] # [48, 16, 16, 16, 16]
    
        out_channels_list = [*out_channels_list, 1] # [16, 16, 16, 16, 1]
        
        #计算每一层神经元的数量
        num_neurons_list = list(map(
            operator.add,
            out_channels_list,
            map(operator.mul, in_channels_list, out_channels_list),
        )) # [784, 272, 272, 272, 17], where 784 is the 48 x 16 + 16
    

        hyper_in_channels_list = [hyper_in_channels, *hyper_out_channels_list] #[256, 256, 256, 256, 256]
        hyper_out_channels_list = [*hyper_out_channels_list, sum(num_neurons_list)] # [256, 256, 256, 256, 1617]
        
        # 定义超网络结构
        self.hypernetwork = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                    nn.LayerNorm(out_channels),
                    nn.GELU(),
                )
                for in_channels, out_channels
                in zip(hyper_in_channels_list[:-1], hyper_out_channels_list[:-1])
            ],
            *[
                nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                )
                for in_channels, out_channels
                in zip(hyper_in_channels_list[-1:], hyper_out_channels_list[-1:])
            ],
        )
        


        self.num_neurons_list = num_neurons_list
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list

        # weight normalization
        # [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868)
        self.apply(lambda module: nn.utils.weight_norm(module) if isinstance(module, nn.Linear) else module)

    def distance_field(self, weights, positions):
        features = positions
        # 对每一层，进行归一化、激活和线性变换
        for layer_index, (weights, in_channels, out_channels) in enumerate(zip(
            torch.split(weights, self.num_neurons_list, dim=-1),
            self.in_channels_list,
            self.out_channels_list,
        )):
            if layer_index:
                features = nn.functional.layer_norm(features, [in_channels])
                features = nn.functional.gelu(features)
            features = torch.einsum(
                "...mn,...n->...m",
                weights.unflatten(-1, (out_channels, in_channels + 1)),
                nn.functional.pad(features, (0, 1), mode="constant", value=1.0),
            )
        distances = features
        return distances

    def forward(self, embeddings):
        weights = self.hypernetwork(embeddings)
        return weights


if __name__=="__main__":

    hyper_distance_field = HyperDistanceField(in_channels=48,
                                                      out_channels_list=[16,16,16,16],
                                                      hyper_in_channels=256,
                                                      hyper_out_channels_list=[256,256,256,256])
    pass