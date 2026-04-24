import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# Assuming layers.py contains ConvBlock1x1 and the required layers
# Assuming .GMNet contains the 'warp' function
from layers import ConvBlock1x1
from .GMNet import warp # Must ensure 'warp' is callable

# --- Auxiliary Class: Positional Encoding Generator ---
class HarmonicProjector(object):
    """
    Generates a positional encoding function for 2D inputs (like flow vectors) 
    by projecting them onto a set of sine and cosine frequencies.
    """
    def __init__(self, **settings):
        self.settings = settings
        self._construct_projection_map()

    def _construct_projection_map(self):
        projection_funcs = []
        input_dim = self.settings["input_dims"]
        total_output_dim = 0

        # 1. Include the raw input (Identity)
        if self.settings["include_input"]:
            projection_funcs.append(lambda x: x)
            total_output_dim += input_dim

        max_log_freq = self.settings["max_freq_log2"]
        num_frequencies = self.settings["num_freqs"]

        # 2. Determine frequency bands
        if self.settings["log_sampling"]:
            frequency_bands = 2.**torch.linspace(0., max_log_freq, steps=num_frequencies)
        else:
            frequency_bands = torch.linspace(2.**0., 2.**max_log_freq, steps=num_frequencies)

        # 3. Create sine/cosine projection functions
        for freq in frequency_bands:
            for p_fn in self.settings["periodic_fns"]:
                # 修复：通过默认参数捕获循环变量，防止其指向循环结束时的最终值
                projection_funcs.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                total_output_dim += input_dim

        self.projection_fns = projection_funcs
        self.output_dimension = total_output_dim

    def project(self, inputs):
        """Projects the input (flow) tensor into the high-dimensional space."""
        return torch.cat([fn(inputs) for fn in self.projection_fns], 1)

# --- Main Class: Feature Integration Module ---
class TemporalIntegrator(nn.Module):
    """
    Module responsible for warping, embedding, and fusing multi-scale features 
    from three consecutive frames (t-1, t, t+1).
    """
    def __init__(self, settings, input_channels, projection_multires=10):
        super(TemporalIntegrator, self).__init__()

        # --- 1. Initialization of Auxiliary Components ---
        projection_settings = {
            "include_input": True,
            "input_dims": 2,
            "max_freq_log2": projection_multires - 1,
            "num_freqs": projection_multires,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }
        self.proj_generator = HarmonicProjector(**projection_settings)
        self.projection_dim = self.proj_generator.output_dimension
        self.feature_dims = input_channels
        self.model_type = settings.backbone
        
        # --- 2. Definition of Fusion Convolution Layers ---
        channel_adaptor_layers = OrderedDict()
        
        # Iterate over feature scales (from coarsest to finest)
        for scale_idx in range(len(input_channels) - 1, -1, -1):
            # Input channels are: (t feature + t flow embedding) + (blended ref feature + blended ref flow embedding)
            # Total input: 2 * (C_enc[i] + C_proj)
            input_ch = 2 * (input_channels[scale_idx] + self.projection_dim)
            output_ch = input_channels[scale_idx]
            channel_adaptor_layers[("scale_adapter", scale_idx)] = ConvBlock1x1(input_ch, output_ch)
        
        self.channel_adaptors = nn.ModuleList(list(channel_adaptor_layers.values()))

    # --- Helper Methods ---

    def _generate_multi_scale_projections(self, motion_vector_field):
        """Downsamples the input motion field and computes positional encodings at each scale."""
        projection_outputs = []
        flow_downsampled = motion_vector_field # Start with original flow
        
        # Calculate embeddings for all feature scales (coarsest first)
        for i in range(len(self.feature_dims)):
            # Downsample flow map
            flow_downsampled = F.interpolate(flow_downsampled, scale_factor=0.5, mode='bilinear', align_corners=False)
            
            # Scale the flow vector values
            flow_downsampled[:, 0, :, :] *= 0.5
            flow_downsampled[:, 1, :, :] *= 0.5
            
            # Special handling for the finest scale if backbone is LSM (matches original logic)
            if i == 0 and self.model_type == "LSM":
                flow_downsampled = F.interpolate(flow_downsampled, scale_factor=0.5, mode='bilinear', align_corners=False)
                flow_downsampled[:, 0, :, :] *= 0.5
                flow_downsampled[:, 1, :, :] *= 0.5
            
            # Project flow vector (Harmonic Projector / Embedder)
            projected_flow = self.proj_generator.project(flow_downsampled)
            projection_outputs.append(projected_flow)

        return projection_outputs

    def _displace_features(self, feature_tensors, vector_field):
        """Warps a list of feature tensors using the provided vector field."""
        warped_tensors = []
        
        for feature_map in feature_tensors:
            _, _, flow_h, flow_w = vector_field.shape
            _, _, H, W = feature_map.shape
            
            # 1. Resize flow field to match feature map size
            flow_resized = F.interpolate(vector_field, size=(H, W), mode="bilinear", align_corners=False)
            
            # 2. Scale flow vectors (from original size to feature map size)
            flow_resized[:, 0, :, :] *= (W / flow_w)
            flow_resized[:, 1, :, :] *= (H / flow_h)
            
            # 3. Perform the warping operation (using external 'warp' function)
            displaced_feature = warp(feature_map, flow_resized)
            warped_tensors.append(displaced_feature)
            
        return warped_tensors

    def _interlace_and_fuse_spatial(self, feature_sets, weighting_mask):
        """
        Performs masked blending of reference features and concatenates the 
        blended result with the current frame feature at each scale.
        
        Input: [Features(t-1), Features(t), Features(t+1)] - all are flow-embedded.
        Output: List of fused (4x channel concatenated) features.
        """
        feats_ref_n1, feats_curr_0, feats_ref_p1 = feature_sets
        fused_outputs = []

        # Iterate over all scales
        for i in range(len(self.feature_dims)):
            feat_n1, feat_0, feat_p1 = feats_ref_n1[i], feats_curr_0[i], feats_ref_p1[i]
            _, _, H, W = feat_0.shape
            
            # 1. Resize mask to match current feature scale
            mask_resized = F.interpolate(weighting_mask, size=(H, W), mode="bilinear", align_corners=False)
            
            # 2. Masked Blending (Blending the flow-embedded reference features)
            # feat = mask * feat_n1 + (1 - mask) * feat_p1
            blended_ref_feat = mask_resized * feat_n1 + (1 - mask_resized) * feat_p1
            
            # 3. Final Concatenation (Concatenating the current frame feature with the blended reference feature)
            fused_outputs.append(torch.cat([feat_0, blended_ref_feat], dim=1))

        return fused_outputs

    # --- Forward Pass ---
    def forward(self, features_triplet, motion_fields, confidence_mask):
        """
        Args:
            features_triplet (list of list of Tensors): Features for [t-1, t, t+1].
            motion_fields (list of Tensors): Flow from t to t-1 and t to t+1. [flow_0_n1, flow_0_p1].
            confidence_mask (Tensor): Mask for merging the reference features.
        """
        feats_n1, feats_0, feats_p1 = features_triplet
        flow_0_n1, flow_0_p1 = motion_fields

        # 1. Warp reference features to current time 't'
        warped_feats_n1 = self._displace_features(feats_n1, flow_0_n1)
        warped_feats_p1 = self._displace_features(feats_p1, flow_0_p1)
        
        # 2. Generate multi-scale positional embeddings for flows (including zero flow)
        zero_flow = 0. * flow_0_n1.clone().detach() # Create zero flow tensor
        
        proj_flow_0 = self._generate_multi_scale_projections(zero_flow)
        proj_flow_n1 = self._generate_multi_scale_projections(flow_0_n1)
        proj_flow_p1 = self._generate_multi_scale_projections(flow_0_p1)

        # 3. Concatenate feature maps with their respective positional embeddings
        flow_embedded_feats_0 = [0 for _ in range(len(feats_0))]
        flow_embedded_feats_n1 = [0 for _ in range(len(feats_0))]
        flow_embedded_feats_p1 = [0 for _ in range(len(feats_0))]

        for i in range(len(feats_0)):
            # Current frame: Feature t + Zero Flow Embedding
            flow_embedded_feats_0[i] = torch.cat([feats_0[i], proj_flow_0[i]], 1)
            # Reference frame t-1: Warped Feature t-1 + Flow t->t-1 Embedding
            flow_embedded_feats_n1[i] = torch.cat([warped_feats_n1[i], proj_flow_n1[i]], 1)
            # Reference frame t+1: Warped Feature t+1 + Flow t->t+1 Embedding
            flow_embedded_feats_p1[i] = torch.cat([warped_feats_p1[i], proj_flow_p1[i]], 1)

        all_embedded_feats = [flow_embedded_feats_n1, flow_embedded_feats_0, flow_embedded_feats_p1]
        
        # 4. Merge the three embedded feature sets using the mask
        fused_tensors = self._interlace_and_fuse_spatial(all_embedded_feats, confidence_mask)

        # 5. Apply 1x1 convolutions to reduce channel dimensionality after concatenation
        for i in range(len(fused_tensors)):
            fused_tensors[i] = self.channel_adaptors[i](fused_tensors[i])

        return fused_tensors
