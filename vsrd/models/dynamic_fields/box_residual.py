
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def rotation_matrix_y(cos, sin):
    one = torch.ones_like(cos)
    zero = torch.zeros_like(cos)
    rotation_matrices = torch.stack([
        torch.stack([ cos, zero,  sin], dim=-1),
        torch.stack([zero,  one, zero], dim=-1),
        torch.stack([-sin, zero,  cos], dim=-1),
    ], dim=-2)
    return rotation_matrices

class BoxParameters3DRBN(nn.Module):
    def __init__(
        self,
        batch_size,
        num_instances,
        num_features=256,
        # x,y,z
        location_range=[
            [-50.0, 1.55 - 1.75 / 2.0 - 5.0, 000.0],
            [+50.0, 1.55 - 1.75 / 2.0 + 5.0, 100.0],
        ],
        # here is the half location
        dimension_range=[
            [0.75, 0.75, 1.5],
            [1.00, 1.00, 2.5],
        ],
    ):
        super().__init__()
        # 7-DOF : # Locations:(N,3)
        # dimensoion:(N,3)
        # orientations: (N,1)
        # instance embeddings: (N,1)
        self.register_parameter(
            "locations",
            nn.Parameter(torch.zeros(batch_size, num_instances, 3)),
        )  # (B,N,3)
        self.register_parameter(
            "dimensions",
            nn.Parameter(torch.zeros(batch_size, num_instances, 3)),
        )
        self.register_parameter(
            "orientations",
            nn.Parameter(torch.tensor([1.0, 0.0]).repeat(batch_size, num_instances, 1)),
        )
        self.register_parameter(
            "embeddings",
            nn.Parameter(torch.rand(num_features).repeat(batch_size, num_instances, 1)),
        )

        self.register_buffer(
            "location_range",
            torch.as_tensor(location_range),
        )
        self.register_buffer(
            "dimension_range",
            torch.as_tensor(dimension_range),
        )

        
        # if self.use_dynamic_field:
        #     embedding_dim = 256
        #     self.positional_embedding = SinusoidalPositionalEmbedding(embedding_dim)
        

    def decode_location(self, locations):
        # 进行线性插值 
        locations = torch.lerp(*self.location_range, torch.sigmoid(locations))
        return locations

    def decode_dimension(self, dimensions):
        dimensions = torch.lerp(*self.dimension_range, torch.sigmoid(dimensions))
        return dimensions

    def decode_orientation(self, orientations):
        orientations = nn.functional.normalize(orientations, dim=-1)
        rotation_matrices = rotation_matrix_y(*torch.unbind(orientations, dim=-1))
        return rotation_matrices

    @staticmethod
    def decode_box_3d(locations, dimensions, orientations,residual=None):
        # NOTE: use the KITTI-360 "evaluation" format instaed of the KITTI-360 "annotation" format
        # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/prepare_train_val_windows.py#L133
        # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/evalDetection.py#L552
        
        # print(locations.shape) #(1,N,3)
        
        if residual is not None:
            locations = locations + residual
        
        
        boxes = dimensions.new_tensor([
            [-1.0, -1.0, +1.0],
            [+1.0, -1.0, +1.0],
            [+1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, +1.0, +1.0],
            [+1.0, +1.0, +1.0],
            [+1.0, +1.0, -1.0],
            [-1.0, +1.0, -1.0],
        ]) * dimensions.unsqueeze(-2) #(1,4,8,3)
        
        boxes = boxes @ orientations.transpose(-2, -1)# make rotation matrix
        boxes = boxes + locations.unsqueeze(-2)
        
        return boxes

    @staticmethod
    def encode_box_3d(boxes_3d):
        
        # (N,4,3)
        locations = torch.mean(boxes_3d, dim=-2)

        widths = torch.mean(torch.norm(torch.sub(
            boxes_3d[..., [1, 2, 6, 5], :],
            boxes_3d[..., [0, 3, 7, 4], :],
        ), dim=-1), dim=-1)

        heights = torch.mean(torch.norm(torch.sub(
            boxes_3d[..., [4, 5, 6, 7], :],
            boxes_3d[..., [0, 1, 2, 3], :],
        ), dim=-1), dim=-1)

        lengths = torch.mean(torch.norm(torch.sub(
            boxes_3d[..., [1, 0, 4, 5], :],
            boxes_3d[..., [2, 3, 7, 6], :],
        ), dim=-1), dim=-1)

        dimensions = torch.stack([widths, heights, lengths], dim=-1) / 2.0

        orientations = torch.mean(torch.sub(
            boxes_3d[..., [1, 0, 4, 5], :],
            boxes_3d[..., [2, 3, 7, 6], :],
        ), dim=-2)

        orientations = nn.functional.normalize(orientations[..., [2, 0]], dim=-1)
        orientations = rotation_matrix_y(*torch.unbind(orientations, dim=-1))

        return locations, dimensions, orientations

    def forward(self):

        # decode box parameters
        locations = self.decode_location(self.locations) #(1,4,3)
        dimensions = self.decode_dimension(self.dimensions)
        orientations = self.decode_orientation(self.orientations) #(N,num_instance,3,3), the final is 3x3 is the rotation matrix.

        
        # if self.use_dynamic_field:
        #     pos_embedding = self.positional_embedding(relative_frame)
        #     nums_of_instances = locations.shape[1]
        #     pos_embedding = pos_embedding.repeat(nums_of_instances,1)
        #     print(pos_embedding.shape)
    
        

        # decode 3D bounding box
        boxes_3d = self.decode_box_3d(
            locations=locations,
            dimensions=dimensions,
            orientations=orientations,
        )

        outputs = dict(
            boxes_3d=boxes_3d,
            locations=locations,
            dimensions=dimensions,
            orientations=orientations,
            embeddings=self.embeddings,
        )

        return outputs

class ResidualBoxPredictor(nn.Module):
    def __init__(self, input_dim=512, output_dim=3, hidden_dims=[256, 128, 64],):
        super(ResidualBoxPredictor, self).__init__()

        embedding_dim = 256
        self.positional_embedding = SinusoidalPositionalEmbedding(embedding_dim)

        # 定义MLP层
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, time, instance_embedding):
        '''
        returns (1,nums_of_instances,nums_of_views,3)
        
        '''
        
        time_embedding = self.positional_embedding(time)
        nums_of_views = time_embedding.shape[0]
        instance_embedding = instance_embedding.unsqueeze(2).repeat(1,1,nums_of_views,1)
        
        nums_of_instances = instance_embedding.shape[1]
        time_embedding = time_embedding.repeat(nums_of_instances,1,1)
        

        
        time_embedding = time_embedding.view(-1,256)
        instance_embedding = instance_embedding.view(-1,256)


    
        # 融合时间嵌入和实例嵌入
        x = torch.cat((time_embedding, instance_embedding), dim=1)
        
        # 通过MLP层
        residual_box = self.mlp(x)
        
        residual_box = residual_box.reshape(1,nums_of_instances,nums_of_views,3)
        
        return residual_box


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.dim = dim

        # Create the positional encodings matrix
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.long)
        return self.pe[x]


if __name__=="__main__":
    
    detector = BoxParameters3DRBN(batch_size=1,
                               num_instances=4,
                               num_features=256)
    
    relative_index = [-1,2,31,41,5,4,10,12]
    

    
    residual_predictor = ResidualBoxPredictor()
    
    residual_box = residual_predictor(time = relative_index, instance_embedding=detector.embeddings)
    

    print(residual_box.shape)
    

    