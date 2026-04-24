import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

# --- A. Custom ResNet Base for Multi-Channel Input ---

class MultichannelCNNBase(models.ResNet):
    """
    Inherits from torchvision's ResNet and modifies the initial convolution layer
    to accept an arbitrary number of input images (N * 3 channels).
    (Equivalent to original ResNetMultiImageInput)
    """
    def __init__(self, layer_block_type, layer_counts, output_classes=1000, input_frame_count=1):
        # Initialize standard ResNet structure
        super(MultichannelCNNBase, self).__init__(layer_block_type, layer_counts)
        
        # Override the first convolution layer (C1)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            input_frame_count * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Standard ResNet initialization of remaining layers and weights
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(layer_block_type, 64, layer_counts[0])
        self.layer2 = self._make_layer(layer_block_type, 128, layer_counts[1], stride=2)
        self.layer3 = self._make_layer(layer_block_type, 256, layer_counts[2], stride=2)
        self.layer4 = self._make_layer(layer_block_type, 512, layer_counts[3], stride=2)

        # Weight initialization (same as original)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def create_multiview_backbone(depth_layers, load_weights=False, num_input_images=1):
    """
    Factory function to instantiate the MultichannelCNNBase with selected depth (18 or 50).
    (Equivalent to original resnet_multiimage_input)
    """
    assert depth_layers in [18, 50], "Only 18 or 50 layer depths are supported for the backbone."
    
    layer_config = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[depth_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[depth_layers]
    
    backbone_instance = MultichannelCNNBase(block_type, layer_config, input_frame_count=num_input_images)

    if load_weights:
        # Load ImageNet weights and adapt the first conv layer
        loaded_weights = model_zoo.load_url(models.resnet.model_urls[f'resnet{depth_layers}'])
        
        # Copy original conv1 weights N times and average (to maintain mean output magnitude)
        input_w = loaded_weights['conv1.weight']
        loaded_weights['conv1.weight'] = torch.cat(
            [input_w] * num_input_images, 1) / num_input_images
            
        backbone_instance.load_state_dict(loaded_weights)
        
    return backbone_instance


# --- B. Feature Extraction Wrapper ---

class MultiviewFeatureExtractor(nn.Module):
    """
    A unified wrapper for extracting multi-scale feature hierarchy using a ResNet backbone.
    Handles standard image normalization prior to feature extraction.
    (Equivalent to original ResnetEncoder)
    """
    def __init__(self, layer_depth, pretrain_status, input_image_count=1):
        super(MultiviewFeatureExtractor, self).__init__()

        # Define the expected channel dimensions for the feature hierarchy outputs (C1, L1, L2, L3, L4)
        self.output_channel_dims = np.array([64, 64, 128, 256, 512])

        base_resnets = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50,
                        101: models.resnet101, 152: models.resnet152}

        if layer_depth not in base_resnets:
            raise ValueError(f"Feature depth {layer_depth} is not a valid ResNet configuration.")

        # Instantiate the backbone, using the custom version if multi-image input is required
        if input_image_count > 1:
            self.backbone = create_multiview_backbone(layer_depth, pretrain_status, input_image_count)
        else:
            self.backbone = base_resnets[layer_depth](pretrain_status)
        
        # Adjust channel dimensions for deep (Bottleneck) ResNets (50, 101, 152)
        if layer_depth > 34:
            self.output_channel_dims[1:] *= 4


    def forward(self, input_frame):
        # ImageNet normalization: x = (input_image - 0.45) / 0.225
        normalized_input = (input_frame - 0.45) / 0.225
        
        hierarchy_outputs = []
        
        # Stage 0: C1 output (post-ReLU)
        x = self.backbone.conv1(normalized_input)
        x = self.backbone.bn1(x)
        hierarchy_outputs.append(self.backbone.relu(x)) # Layer index 0

        # Stage 1: L1 output (after MaxPool)
        x = self.backbone.layer1(self.backbone.maxpool(hierarchy_outputs[-1]))
        hierarchy_outputs.append(x) # Layer index 1
        
        # Stage 2: L2 output
        x = self.backbone.layer2(hierarchy_outputs[-1])
        hierarchy_outputs.append(x) # Layer index 2
        
        # Stage 3: L3 output
        x = self.backbone.layer3(hierarchy_outputs[-1])
        hierarchy_outputs.append(x) # Layer index 3
        
        # Stage 4: L4 output
        x = self.backbone.layer4(hierarchy_outputs[-1])
        hierarchy_outputs.append(x) # Layer index 4

        return hierarchy_outputs


# --- C. Camera Motion Regression Head ---

class MotionRegressionHead(nn.Module):
    """
    Network head that regresses relative camera pose (rotation and translation) 
    from the concatenated L4 feature maps of the input frames.
    (Equivalent to original PoseDecoder)
    """
    def __init__(self, input_channel_dims, num_feats_in, frames_for_pred=None, downsample_stride=1):
        super(MotionRegressionHead, self).__init__()

        self.input_channel_dims = input_channel_dims
        self.feature_count = num_feats_in

        # Default prediction target count is num_feats_in - 1 (predicting relative pose for all pairs)
        if frames_for_pred is None:
            frames_for_pred = num_feats_in - 1
        self.prediction_target_count = frames_for_pred

        self.conv_sequence = OrderedDict()
        
        # 1. Dimensionality Reduction (Squeeze) on L4 features
        self.conv_sequence[("dimensionality_reduction")] = nn.Conv2d(self.input_channel_dims[-1], 256, 1)
        
        # 2. Sequential Regression Blocks
        self.conv_sequence[("pose_block", 0)] = nn.Conv2d(self.feature_count * 256, 256, 3, downsample_stride, 1)
        self.conv_sequence[("pose_block", 1)] = nn.Conv2d(256, 256, 3, downsample_stride, 1)
        
        # 3. Final Prediction (6DOF: 3 rotation, 3 translation)
        self.conv_sequence[("pose_block", 2)] = nn.Conv2d(256, 6 * self.prediction_target_count, 1)

        self.activation = nn.ReLU()
        self.network_layers = nn.ModuleList(list(self.conv_sequence.values()))


    def forward(self, input_feature_sets):
        """
        Args:
            input_feature_sets (list of lists): List of feature hierarchy outputs for each input frame.
                                                We only use the last feature map (L4/index -1).
        """
        # Extract L4 feature map for all input frames
        last_layer_features = [f[-1] for f in input_feature_sets]

        # 1. Apply Dimensionality Reduction and ReLU
        cat_features = [self.activation(self.conv_sequence["dimensionality_reduction"](f)) for f in last_layer_features]
        # Concatenate features from all frames along the channel dimension
        cat_features = torch.cat(cat_features, 1)

        # 2. Run through sequential blocks
        output_tensor = cat_features
        for i in range(3):
            output_tensor = self.conv_sequence[("pose_block", i)](output_tensor)
            if i != 2:
                output_tensor = self.activation(output_tensor)

        # 3. Global Spatial Averaging (2D Mean Pooling)
        output_tensor = output_tensor.mean(3).mean(2)

        # 4. Scale and Reshape to (Batch, Num_Frames_to_Predict, 1, 6)
        # Original scaling factor is 0.01
        output_tensor = 0.01 * output_tensor.view(-1, self.prediction_target_count, 1, 6)

        # 5. Split 6DOF output into rotation (axis-angle) and translation vectors
        rotation_params = output_tensor[..., :3]
        translation_vector = output_tensor[..., 3:]

        return rotation_params, translation_vector
