import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

# --- I. Auxiliary Functions ---

def spatial_sampler(source_image, displacement_field):
    """
    Performs bilinear sampling (warping) of a source image using a 2D displacement field.
    This replaces the original 'warp' function.
    """
    B, _, H, W = displacement_field.shape
    
    # 1. Create normalized coordinate grid (from -1.0 to 1.0)
    x_coords = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    y_coords = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    normalized_grid = torch.cat([x_coords, y_coords], 1).to(source_image)
    
    # 2. Normalize the flow vectors to [-2, 2] range (matching normalized grid)
    normalized_flow = torch.cat([
        displacement_field[:, 0:1, :, :] / ((W - 1.0) / 2.0), 
        displacement_field[:, 1:2, :, :] / ((H - 1.0) / 2.0)
    ], 1)
    
    # 3. Compute the sampling grid (normalized coordinates + normalized flow)
    # Resulting shape: (B, H, W, 2)
    sampling_grid = (normalized_grid + normalized_flow).permute(0, 2, 3, 1)
    
    # 4. Perform grid sampling (Bilinear interpolation)
    output_image = F.grid_sample(input=source_image, grid=sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    return output_image


def compute_normalized_robust_weight(prediction, ground_truth, damping_factor):
    """Calculates robust weight based on End-Point Error (EPE) using a Lorentzian kernel."""
    epe = ((prediction.detach() - ground_truth) ** 2).sum(dim=1, keepdim=True) ** 0.5
    robust_weight = torch.exp(-damping_factor * epe)
    return robust_weight


def feature_upsample(tensor, ratio):
    """Resizes a tensor using bilinear interpolation."""
    return F.interpolate(tensor, scale_factor=ratio, mode="bilinear", align_corners=False)


def residual_conv_prelu(in_dim, out_dim, k=3, s=1, p=1, d=1, g=1, bias=True):
    """Sequential Conv + PReLU block."""
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, k, s, p, d, g, bias=bias), 
        nn.PReLU(out_dim)
    )

# --- II. Network Building Blocks ---

class ResidualSplitBlock(nn.Module):
    """
    A residual block with split-channel processing (ResBlock in original).
    The side_channels part is processed separately.
    """
    def __init__(self, full_channels, side_channels_dim, use_bias=True):
        super(ResidualSplitBlock, self).__init__()
        self.side_dim = side_channels_dim
        
        # Main path convs (operate on full full_channels)
        self.main_conv_A = residual_conv_prelu(full_channels, full_channels, bias=use_bias)
        self.main_conv_B = residual_conv_prelu(full_channels, full_channels, bias=use_bias)
        
        # Side path convs (operate only on side_channels_dim)
        self.side_conv_A = residual_conv_prelu(side_channels_dim, side_channels_dim, bias=use_bias)
        self.side_conv_B = residual_conv_prelu(side_channels_dim, side_channels_dim, bias=use_bias)

        # Final components
        self.final_conv = nn.Conv2d(full_channels, full_channels, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.final_activation = nn.PReLU(full_channels)

    def forward(self, x):
        # 1. First main convolution
        out = self.main_conv_A(x)
        
        # 2. Extract and process side channels
        full_ch = out.size(1)
        # Identify start index for side channels
        start_idx = full_ch - self.side_dim 
        
        side_input = out[:, start_idx:, :, :].clone()
        side_output = self.side_conv_A(side_input)
        out[:, start_idx:, :, :] = side_output
        
        # 3. Second main convolution
        out = self.main_conv_B(out)
        
        # 4. Extract and process side channels again
        side_input_B = out[:, start_idx:, :, :].clone()
        side_output_B = self.side_conv_B(side_input_B)
        out[:, start_idx:, :, :] = side_output_B
        
        # 5. Final residual connection
        residual_branch = self.final_conv(out)
        return self.final_activation(x + residual_branch)


class PyramidFeatureExtractorL(nn.Module):
    """Encoder_L in original code."""
    def __init__(self):
        super(PyramidFeatureExtractorL, self).__init__()
        self.level_1 = nn.Sequential(
            residual_conv_prelu(3, 64, 7, 2, 3), 
            residual_conv_prelu(64, 64, 3, 1, 1)
        )
        self.level_2 = nn.Sequential(
            residual_conv_prelu(64, 96, 3, 2, 1), 
            residual_conv_prelu(96, 96, 3, 1, 1)
        )
        self.level_3 = nn.Sequential(
            residual_conv_prelu(96, 144, 3, 2, 1), 
            residual_conv_prelu(144, 144, 3, 1, 1)
        )
        self.level_4 = nn.Sequential(
            residual_conv_prelu(144, 192, 3, 2, 1), 
            residual_conv_prelu(192, 192, 3, 1, 1)
        )
        
    def forward(self, image):
        f1 = self.level_1(image)
        f2 = self.level_2(f1)
        f3 = self.level_3(f2)
        f4 = self.level_4(f3)
        return f1, f2, f3, f4


class PyramidFeatureExtractorS(nn.Module):
    """Encoder_S in original code."""
    def __init__(self):
        super(PyramidFeatureExtractorS, self).__init__()
        self.level_1 = nn.Sequential(
            residual_conv_prelu(3, 24, 3, 2, 1), 
            residual_conv_prelu(24, 24, 3, 1, 1)
        )
        self.level_2 = nn.Sequential(
            residual_conv_prelu(24, 36, 3, 2, 1), 
            residual_conv_prelu(36, 36, 3, 1, 1)
        )
        self.level_3 = nn.Sequential(
            residual_conv_prelu(36, 54, 3, 2, 1), 
            residual_conv_prelu(54, 54, 3, 1, 1)
        )
        self.level_4 = nn.Sequential(
            residual_conv_prelu(54, 72, 3, 2, 1), 
            residual_conv_prelu(72, 72, 3, 1, 1)
        )
        
    def forward(self, image):
        f1 = self.level_1(image)
        f2 = self.level_2(f1)
        f3 = self.level_3(f2)
        f4 = self.level_4(f3)
        return f1, f2, f3, f4


class HierarchicalRefiner4L(nn.Module):
    """Decoder4_L in original code."""
    def __init__(self):
        super(HierarchicalRefiner4L, self).__init__()
        self.processing_path = nn.Sequential(
            residual_conv_prelu(384 + 1, 384), # f0, f1, embt
            ResidualSplitBlock(384, 64), 
            nn.ConvTranspose2d(384, 148, 4, 2, 1, bias=True)
        )
        
    def forward(self, feat_curr, feat_ref, embedding_t):
        B, C, H, W = feat_curr.shape
        # Repeat embedding across spatial dims
        embedding_t_spatial = embedding_t.repeat(1, 1, H, W)
        
        f_in = torch.cat([feat_curr, feat_ref, embedding_t_spatial], 1)
        f_out = self.processing_path(f_in)
        return f_out


class HierarchicalRefiner3L(nn.Module):
    """Decoder3_L in original code."""
    def __init__(self):
        super(HierarchicalRefiner3L, self).__init__()
        self.processing_path = nn.Sequential(
            residual_conv_prelu(436, 432), 
            ResidualSplitBlock(432, 64), 
            nn.ConvTranspose2d(432, 100, 4, 2, 1, bias=True)
        )

    def forward(self, feat_t_up, feat_0, feat_1, up_flow_0_prev, up_flow_1_prev):
        feat_0_warped = spatial_sampler(feat_0, up_flow_0_prev)
        feat_1_warped = spatial_sampler(feat_1, up_flow_1_prev)
        # [ft_up, f0_warp, f1_warp, up_flow0_prev, up_flow1_prev]
        f_in = torch.cat([feat_t_up, feat_0_warped, feat_1_warped, up_flow_0_prev, up_flow_1_prev], 1)
        f_out = self.processing_path(f_in)
        return f_out


class HierarchicalRefiner2L(nn.Module):
    """Decoder2_L in original code."""
    def __init__(self):
        super(HierarchicalRefiner2L, self).__init__()
        self.processing_path = nn.Sequential(
            residual_conv_prelu(292, 288), 
            ResidualSplitBlock(288, 64), 
            nn.ConvTranspose2d(288, 68, 4, 2, 1, bias=True)
        )

    def forward(self, feat_t_up, feat_0, feat_1, up_flow_0_prev, up_flow_1_prev):
        feat_0_warped = spatial_sampler(feat_0, up_flow_0_prev)
        feat_1_warped = spatial_sampler(feat_1, up_flow_1_prev)
        # [ft_up, f0_warp, f1_warp, up_flow0_prev, up_flow1_prev]
        f_in = torch.cat([feat_t_up, feat_0_warped, feat_1_warped, up_flow_0_prev, up_flow_1_prev], 1)
        f_out = self.processing_path(f_in)
        return f_out


class HierarchicalRefiner1L(nn.Module):
    """Decoder1_L in original code."""
    def __init__(self):
        super(HierarchicalRefiner1L, self).__init__()
        self.processing_path = nn.Sequential(
            residual_conv_prelu(196, 192), 
            ResidualSplitBlock(192, 64), 
            nn.ConvTranspose2d(192, 8, 4, 2, 1, bias=True)
        )
        
    def forward(self, feat_t_up, feat_0, feat_1, up_flow_0_prev, up_flow_1_prev):
        feat_0_warped = spatial_sampler(feat_0, up_flow_0_prev)
        feat_1_warped = spatial_sampler(feat_1, up_flow_1_prev)
        # [ft_up, f0_warp, f1_warp, up_flow0_prev, up_flow1_prev]
        f_in = torch.cat([feat_t_up, feat_0_warped, feat_1_warped, up_flow_0_prev, up_flow_1_prev], 1)
        f_out = self.processing_path(f_in)
        return f_out


class HierarchicalRefiner4S(nn.Module):
    """Decoder4_S in original code."""
    def __init__(self):
        super(HierarchicalRefiner4S, self).__init__()
        self.processing_path = nn.Sequential(
            residual_conv_prelu(144 + 1, 144), 
            ResidualSplitBlock(144, 24), 
            nn.ConvTranspose2d(144, 58, 4, 2, 1, bias=True)
        )
        
    def forward(self, feat_curr, feat_ref, embedding_t):
        B, C, H, W = feat_curr.shape
        embedding_t_spatial = embedding_t.repeat(1, 1, H, W)
        f_in = torch.cat([feat_curr, feat_ref, embedding_t_spatial], 1)
        f_out = self.processing_path(f_in)
        return f_out


class HierarchicalRefiner3S(nn.Module):
    """Decoder3_S in original code."""
    def __init__(self):
        super(HierarchicalRefiner3S, self).__init__()
        self.processing_path = nn.Sequential(
            residual_conv_prelu(166, 162), 
            ResidualSplitBlock(162, 24), 
            nn.ConvTranspose2d(162, 40, 4, 2, 1, bias=True)
        )

    def forward(self, feat_t_up, feat_0, feat_1, up_flow_0_prev, up_flow_1_prev):
        feat_0_warped = spatial_sampler(feat_0, up_flow_0_prev)
        feat_1_warped = spatial_sampler(feat_1, up_flow_1_prev)
        f_in = torch.cat([feat_t_up, feat_0_warped, feat_1_warped, up_flow_0_prev, up_flow_1_prev], 1)
        f_out = self.processing_path(f_in)
        return f_out


class HierarchicalRefiner2S(nn.Module):
    """Decoder2_S in original code."""
    def __init__(self):
        super(HierarchicalRefiner2S, self).__init__()
        self.processing_path = nn.Sequential(
            residual_conv_prelu(112, 108), 
            ResidualSplitBlock(108, 24), 
            nn.ConvTranspose2d(108, 28, 4, 2, 1, bias=True)
        )

    def forward(self, feat_t_up, feat_0, feat_1, up_flow_0_prev, up_flow_1_prev):
        feat_0_warped = spatial_sampler(feat_0, up_flow_0_prev)
        feat_1_warped = spatial_sampler(feat_1, up_flow_1_prev)
        f_in = torch.cat([feat_t_up, feat_0_warped, feat_1_warped, up_flow_0_prev, up_flow_1_prev], 1)
        f_out = self.processing_path(f_in)
        return f_out


class HierarchicalRefiner1S(nn.Module):
    """Decoder1_S in original code."""
    def __init__(self):
        super(HierarchicalRefiner1S, self).__init__()
        self.processing_path = nn.Sequential(
            residual_conv_prelu(76, 72), 
            ResidualSplitBlock(72, 24), 
            nn.ConvTranspose2d(72, 8, 4, 2, 1, bias=True)
        )
        
    def forward(self, feat_t_up, feat_0, feat_1, up_flow_0_prev, up_flow_1_prev):
        feat_0_warped = spatial_sampler(feat_0, up_flow_0_prev)
        feat_1_warped = spatial_sampler(feat_1, up_flow_1_prev)
        f_in = torch.cat([feat_t_up, feat_0_warped, feat_1_warped, up_flow_0_prev, up_flow_1_prev], 1)
        f_out = self.processing_path(f_in)
        return f_out

# --- III. Loss Functions ---

class TernaryLoss(nn.Module):
    """Ternary loss function (Ternary in original) for texture similarity."""
    def __init__(self, kernel_size=7):
        super(TernaryLoss, self).__init__()
        self.k_size = kernel_size
        out_channels = kernel_size * kernel_size
        # Identity kernel for convolution
        kernel = np.eye(out_channels).reshape((kernel_size, kernel_size, 1, out_channels))
        kernel = np.transpose(kernel, (3, 2, 0, 1))
        self.register_buffer('kernel', torch.tensor(kernel).float())

    def feature_difference(self, tensor):
        kernel = self.kernel.to(tensor.device)
        # Convert to single channel intensity
        tensor_intensity = tensor.mean(dim=1, keepdim=True)
        
        # Convolve to get patches (local neighbors)
        padding = self.k_size // 2
        patches = F.conv2d(tensor_intensity, kernel, padding=padding, bias=None)
        
        # Calculate local difference from center
        local_diff = patches - tensor_intensity
        # Apply normalization: diff / sqrt(0.81 + diff^2)
        normalized_diff = local_diff / torch.sqrt(0.81 + local_diff ** 2)
        return normalized_diff

    def valid_region_mask(self, tensor):
        padding = self.k_size // 2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        return mask
        
    def forward(self, x, y):
        diff_x = self.feature_difference(x)
        diff_y = self.feature_difference(y)
        
        # Compute distance metric
        diff_total = diff_x - diff_y.detach()
        dist = (diff_total ** 2 / (0.1 + diff_total ** 2)).mean(dim=1, keepdim=True)
        
        # Apply valid mask
        mask = self.valid_region_mask(x)
        loss = (dist * mask).mean()
        return loss


class GeometryConsistency(nn.Module):
    """Geometry loss (Geometry in original) based on local patch differences."""
    def __init__(self, kernel_size=3):
        super(GeometryConsistency, self).__init__()
        self.k_size = kernel_size
        out_channels = kernel_size * kernel_size
        # Identity kernel for convolution
        kernel = np.eye(out_channels).reshape((kernel_size, kernel_size, 1, out_channels))
        kernel = np.transpose(kernel, (3, 2, 0, 1))
        self.register_buffer('kernel', torch.tensor(kernel).float())

    def feature_transform(self, tensor):
        kernel = self.kernel.to(tensor.device)
        b, c, h, w = tensor.size()
        
        # Reshape to (B*C, 1, H, W) for single-channel convolution
        tensor_reshaped = tensor.reshape(b * c, 1, h, w)
        padding = self.k_size // 2
        
        # Convolve to get patches (local neighbors)
        patches = F.conv2d(tensor_reshaped, kernel, padding=padding, bias=None)
        
        # Calculate local difference from center
        local_diff = patches - tensor_reshaped
        
        # Reshape back to (B, C * (k^2), H, W)
        local_diff_full = local_diff.reshape(b, c * (self.k_size ** 2), h, w)
        
        # Apply normalization: diff / sqrt(0.81 + diff^2)
        normalized_diff = local_diff_full / torch.sqrt(0.81 + local_diff_full ** 2)
        return normalized_diff

    def valid_region_mask(self, tensor):
        padding = self.k_size // 2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, x, y):
        diff_x = self.feature_transform(x)
        diff_y = self.feature_transform(y)
        
        # Compute distance metric
        diff_total = diff_x - diff_y # Note: No detach on y, matching original Geometry loss
        dist = (diff_total ** 2 / (0.1 + diff_total ** 2)).mean(dim=1, keepdim=True)
        
        # Apply valid mask
        mask = self.valid_region_mask(x)
        loss = (dist * mask).mean()
        return loss


class L1Charbonnier(nn.Module):
    """Charbonnier_L1 in original code."""
    def __init__(self):
        super(L1Charbonnier, self).__init__()

    def forward(self, difference, mask=None):
        eps = 1e-6
        robust_term = (difference ** 2 + eps) ** 0.5
        
        if mask is None:
            loss = robust_term.mean()
        else:
            # Weighted mean
            loss = (robust_term * mask).mean() / (mask.mean() + 1e-9)
        return loss


class AdaptiveCharbonnier(nn.Module):
    """Charbonnier_Ada in original code."""
    def __init__(self):
        super(AdaptiveCharbonnier, self).__init__()

    def forward(self, difference, weight_map):
        # weight_map (w) is used to calculate alpha and epsilon
        alpha = weight_map / 2
        # epsilon calculation matching original logic: 10^(-(10w - 1)/3)
        epsilon_squared = 10 ** (-(10 * weight_map - 1) / 3) 
        epsilon = epsilon_squared ** 0.5 # Calculate epsilon first
        
        # Generalized Charbonnier loss: (diff^2 + eps^2)^alpha
        loss = ((difference ** 2 + epsilon ** 2) ** alpha).mean()
        return loss
    
# --- IV. Main Network (Flow Estimator) ---

class FlowEstimator(nn.Module):
    """
    The main flow estimation network (GMNet in original).
    Predicts bi-directional flow, merge mask, and includes custom losses.
    """
    def __init__(self, scale="large"):
        super(FlowEstimator, self).__init__()
        
        if scale == "large":
            self.encoder_module = PyramidFeatureExtractorL()
            self.decoder_refiner_4 = HierarchicalRefiner4L()
            self.decoder_refiner_3 = HierarchicalRefiner3L()
            self.decoder_refiner_2 = HierarchicalRefiner2L()
            self.decoder_refiner_1 = HierarchicalRefiner1L()
        elif scale == "small":
            self.encoder_module = PyramidFeatureExtractorS()
            self.decoder_refiner_4 = HierarchicalRefiner4S()
            self.decoder_refiner_3 = HierarchicalRefiner3S()
            self.decoder_refiner_2 = HierarchicalRefiner2S()
            self.decoder_refiner_1 = HierarchicalRefiner1S()
        else:
            raise ValueError(f"Unknown scale: {scale}. Choose 'large' or 'small'.")
            
        # Loss components
        self.photometric_loss = L1Charbonnier()
        self.texture_loss = TernaryLoss(7)
        self.robust_loss = AdaptiveCharbonnier()
        self.geometric_loss = GeometryConsistency(3)
    

    def forward(self, image_0, image_1, embedding_t, target_image=None, resolution_factors=(1.0, 0.5), flow_only=False):
        
        _, _, H, W = image_0.shape
        
        # Adjustment for specific KITTI resolution
        if H == 320 and W == 1024:
            resolution_factors = (0.6, 0.3125)

        # 1. Image Preprocessing (mean centering)
        image_pair = torch.cat([image_0, image_1], 2)
        mean_intensity = image_pair.mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        
        img0_centered = image_0 - mean_intensity
        img1_centered = image_1 - mean_intensity

        # 2. Downsample for Flow Estimation
        fh, fw = int(H * resolution_factors[0]), int(W * resolution_factors[1])
        img0_downscaled = F.interpolate(img0_centered, size=(fh, fw), mode="bilinear", align_corners=False)
        img1_downscaled = F.interpolate(img1_centered, size=(fh, fw), mode="bilinear", align_corners=False)

        # 3. Feature Extraction
        f0_1, f0_2, f0_3, f0_4 = self.encoder_module(img0_downscaled)
        f1_1, f1_2, f1_3, f1_4 = self.encoder_module(img1_downscaled)

        # If target image (imgt) is provided (for loss calculation), extract its features too
        ft_1, ft_2, ft_3, ft_4 = None, None, None, None
        if target_image is not None:
            target_image_centered = target_image - mean_intensity
            target_image_downscaled = F.interpolate(target_image_centered, size=(fh, fw), mode="bilinear", align_corners=False)
            ft_1, ft_2, ft_3, ft_4 = self.encoder_module(target_image_downscaled)

        # 4. Decoding and Upsampling (Hierarchical Refinement)

        # Level 4
        out4 = self.decoder_refiner_4(f0_4, f1_4, embedding_t)
        flow_0_4 = out4[:, 0:2]
        flow_1_4 = out4[:, 2:4]
        feat_t_3_up = out4[:, 4:] # Feature for next level

        # Level 3
        out3 = self.decoder_refiner_3(feat_t_3_up, f0_3, f1_3, flow_0_4, flow_1_4)
        # Residual update: upsampled_prev_flow * 2.0 + current_prediction
        flow_0_3 = out3[:, 0:2] + 2.0 * feature_upsample(flow_0_4, 2.0)
        flow_1_3 = out3[:, 2:4] + 2.0 * feature_upsample(flow_1_4, 2.0)
        feat_t_2_up = out3[:, 4:]

        # Level 2
        out2 = self.decoder_refiner_2(feat_t_2_up, f0_2, f1_2, flow_0_3, flow_1_3)
        flow_0_2 = out2[:, 0:2] + 2.0 * feature_upsample(flow_0_3, 2.0)
        flow_1_2 = out2[:, 2:4] + 2.0 * feature_upsample(flow_1_3, 2.0)
        feat_t_1_up = out2[:, 4:]

        # Level 1 (Final)
        out1 = self.decoder_refiner_1(feat_t_1_up, f0_1, f1_1, flow_0_2, flow_1_2)
        flow_0_1 = out1[:, 0:2] + 2.0 * feature_upsample(flow_0_2, 2.0)
        flow_1_1 = out1[:, 2:4] + 2.0 * feature_upsample(flow_1_2, 2.0)
        merge_mask_1 = torch.sigmoid(out1[:, 4:5])
        # Original residual component (out1[:, 5:]) is noted as being dropped.

        # 5. Final Upscaling and Normalization
        
        # Flow 0->1
        final_flow_0 = F.interpolate(flow_0_1, size=(H, W), mode="bilinear", align_corners=False)
        final_flow_0[:, 0, :, :] *= (1.0 / resolution_factors[1]) # Rescale X-component
        final_flow_0[:, 1, :, :] *= (1.0 / resolution_factors[0]) # Rescale Y-component
        
        # Flow 1->0
        final_flow_1 = F.interpolate(flow_1_1, size=(H, W), mode="bilinear", align_corners=False)
        final_flow_1[:, 0, :, :] *= (1.0 / resolution_factors[1]) # Rescale X-component
        final_flow_1[:, 1, :, :] *= (1.0 / resolution_factors[0]) # Rescale Y-component
        
        # Merge Mask
        final_merge_mask = F.interpolate(merge_mask_1, size=(H, W), mode="bilinear", align_corners=False)

        if flow_only:
            return final_flow_0, final_flow_1, final_merge_mask
        
        # 6. View Synthesis and Output Generation
        
        img0_warped = spatial_sampler(image_0, final_flow_0)
        img1_warped = spatial_sampler(image_1, final_flow_1)
        
        # Merged/Reconstructed image (without residual component)
        merged_image_centered = final_merge_mask * img0_warped + (1 - final_merge_mask) * img1_warped
        
        # Add back mean and clamp
        predicted_image = merged_image_centered + mean_intensity
        predicted_image = torch.clamp(predicted_image, 0, 1)

        # 7. Loss Calculation
        if target_image is not None:
            # Reconstruction Loss: L1 Charbonnier + Ternary
            loss_rec = self.photometric_loss(merged_image_centered - target_image_centered) + \
                       self.texture_loss(merged_image_centered, target_image_centered)
            
            # Geometry Consistency Loss (only on internal features)
            loss_geo = 0.01 * (self.geometric_loss(feat_t_1_up, ft_1) + 
                               self.geometric_loss(feat_t_2_up, ft_2) + 
                               self.geometric_loss(feat_t_3_up, ft_3))
            
            total_loss = loss_rec + loss_geo
            
            return predicted_image, total_loss, final_flow_0, final_flow_1, final_merge_mask
        else:
            return predicted_image, final_flow_0, final_flow_1, final_merge_mask
