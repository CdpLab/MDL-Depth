import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import copy
import torch
import torch.nn.functional as F
from torchvision import transforms

# Assuming necessary modules are imported (LSM, GMNet, FusionModule, disp_to_depth, etc.)
from networks import LSM, GMNet, FusionModule, ResnetEncoder, PoseDecoder
from layers import disp_to_depth
# STEREO_SCALE_FACTOR is typically a constant, defined here for completeness if not imported
STEREO_SCALE_FACTOR = 5.4 


def parse_test_configuration():
    """Defines and parses configuration arguments for the video processing pipeline."""
    parser = argparse.ArgumentParser(
        description='Script for testing MDL-Depth models on video sequences.')

    parser.add_argument('--media_source_path', type=str,
                        help='Path to a single image or a folder of images (video sequence)', required=True)
    parser.add_argument('--checkpoint_file_path',
                        type=str,
                        help="Path to the model checkpoint (.pth file)", required=True)
    parser.add_argument('--dataset_origin',
                        type=str,
                        default="kitti",
                        choices=["kitti", "cityscapes"],
                        help="The dataset the model was trained on (used for flow weights).")
    parser.add_argument("--backbone_arch",
                        type=str,
                        default="ResNet18",
                        choices=["ResNet18", "ResNet50", "LSM"],
                        help="The feature extractor architecture used.")
    parser.add_argument("--flow_network_scale",
                        type=str,
                        help="Scale of the fixed flow prediction network (GMNet scale)",
                        default="small",
                        choices=["large", "small"])
    parser.add_argument("--batch_size",
                        type=int,
                        help="Processing batch size",
                        default=4)
    parser.add_argument("--input_resolution_v",
                        type=int,
                        help="Input image height for feeding the network",
                        default=192)
    parser.add_argument("--input_resolution_h",
                        type=int,
                        help="Input image width for feeding the network",
                        default=640)
    parser.add_argument("--depth_min",
                        type=float,
                        help="Minimum valid depth limit",
                        default=0.1)
    parser.add_argument("--depth_max",
                        type=float,
                        help="Maximum valid depth limit",
                        default=100.0)
    parser.add_argument('--image_extension', type=str,
                        help='File extension to search for in the media source folder', default="png")
    parser.add_argument("--disable_gpu",
                        help='If set, forces usage of CPU.',
                        action='store_true')
    parser.add_argument("--save_numpy_data",
                        help='If set, saves raw numpy arrays of predicted disparity/depth.',
                        action='store_true')
    return parser.parse_args()


def load_network_modules(config, device):
    """Initializes and loads the weights into all required neural network modules."""
    print(f"-> Loading model from {config.checkpoint_file_path}")
    
    # Load the full checkpoint dictionary
    checkpoint = torch.load(config.checkpoint_file_path, map_location='cpu')

    # 1. Initialize Depth Encoder (Feature Extractor)
    if config.backbone_arch == "LSM":
        feature_extractor = LSM.DepthEncoder(
            model='lsm', drop_path_rate=0.2,
            width=config.input_resolution_h, height=config.input_resolution_v
        )
        # 2. Initialize Depth Decoders and Fusion Module
        disparity_predictor_mono = LSM.DepthDecoder(feature_extractor.num_ch_enc, range(1))
        disparity_predictor_fused = copy.deepcopy(disparity_predictor_mono)
        temporal_fusion_unit = FusionModule(config, feature_extractor.num_ch_enc)
    else:
        # Placeholder for other backbones (e.g., if it uses ResnetEncoder/Decoder from another file)
        raise NotImplementedError(f"Backbone {config.backbone_arch} not implemented for inference loading.")

    # 3. Load weights into Depth Modules
    feature_extractor.load_state_dict({k: v for k, v in checkpoint["encoder"].items() if k in feature_extractor.state_dict()})
    disparity_predictor_mono.load_state_dict({k: v for k, v in checkpoint["depth"].items() if k in disparity_predictor_mono.state_dict()})
    disparity_predictor_fused.load_state_dict({k: v for k, v in checkpoint["depth_mf"].items() if k in disparity_predictor_fused.state_dict()})
    temporal_fusion_unit.load_state_dict({k: v for k, v in checkpoint["fusion_module"].items() if k in temporal_fusion_unit.state_dict()})

    # 4. Initialize and Load Flow Prediction Engine (GMNet)
    flow_prediction_engine = GMNet(scale=config.flow_network_scale)
    
    flow_weights_map = {
        "kitti": {"large": "./weights/GMNet_L_KITTI.pth", "small": "./weights/GMNet_S_KITTI.pth"},
        "cityscapes": {"large": "./weights/GMNet_L_CS.pth", "small": "./weights/GMNet_S_CS.pth"}
    }
    
    if config.dataset_origin in flow_weights_map:
        weights_path = flow_weights_map[config.dataset_origin][config.flow_network_scale]
        flow_prediction_engine.load_state_dict(torch.load(weights_path)["MDL"])

    # 5. Move all models to device and set to evaluation mode
    feature_extractor.to(device).eval()
    disparity_predictor_mono.to(device).eval()
    disparity_predictor_fused.to(device).eval()
    temporal_fusion_unit.to(device).eval()
    flow_prediction_engine.to(device).eval()
    
    return feature_extractor, disparity_predictor_mono, disparity_predictor_fused, temporal_fusion_unit, flow_prediction_engine


def run_video_pipeline(config):
    """Executes the full video processing pipeline."""
    
    # 1. Setup Device
    if torch.cuda.is_available() and not config.disable_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # 2. Load Models
    (feature_extractor, disparity_predictor_mono, disparity_predictor_fused, 
     temporal_fusion_unit, flow_prediction_engine) = load_network_modules(config, device)

    # 3. Get Video Sequence Files
    source_path = config.media_source_path
    if not os.path.isdir(source_path):
        raise ValueError("Media source path must be a directory containing video frames.")

    video_sequence_files = sorted(glob.glob(os.path.join(source_path, f'*.{config.image_extension}')))
    
    output_directory = source_path
    
    if not video_sequence_files:
        print(f"-> Found no images with extension '{config.image_extension}' in {source_path}")
        return

    print(f"-> Predicting on {len(video_sequence_files)} video frames")
    
    feed_height = config.input_resolution_v
    feed_width = config.input_resolution_h
    
    # 4. Process Each Frame in Sequence
    gif_frames_output = []
    with torch.no_grad():
        
        for index, current_path in enumerate(video_sequence_files):

            # A. Load and Preprocess Current Frame
            original_image_pil = pil.open(current_path).convert('RGB')
            original_image_np = np.array(original_image_pil).copy()
            original_width, original_height = original_image_pil.size
            
            # Resize and tensorize the current image
            resized_current_pil = original_image_pil.resize((feed_width, feed_height), pil.LANCZOS)
            current_frame_tensor = transforms.ToTensor()(resized_current_pil).unsqueeze(0).to(device)

            # B. Load and Preprocess Neighbor Frames (Handling boundary conditions)
            if index == 0:
                neighbor_frame_prev = current_frame_tensor
            else:
                prev_path = video_sequence_files[index - 1]
                neighbor_frame_prev = pil.open(prev_path).convert('RGB').resize((feed_width, feed_height), pil.LANCZOS)
                neighbor_frame_prev = transforms.ToTensor()(neighbor_frame_prev).unsqueeze(0).to(device)
            
            if index == len(video_sequence_files) - 1:
                neighbor_frame_next = current_frame_tensor
            else:
                next_path = video_sequence_files[index + 1]
                neighbor_frame_next = pil.open(next_path).convert('RGB').resize((feed_width, feed_height), pil.LANCZOS)
                neighbor_frame_next = transforms.ToTensor()(neighbor_frame_next).unsqueeze(0).to(device)
                
            # C. --- Single-Frame Prediction ---
            current_features = feature_extractor(current_frame_tensor)
            mono_outputs = disparity_predictor_mono(current_features)

            mono_disparity = mono_outputs[("disp", 0)]
            mono_disparity_resized = F.interpolate(
                mono_disparity, (original_height, original_width), mode="bilinear", align_corners=False)

            # D. --- Multi-Frame (Fused) Prediction ---
            
            # D1. Prepare temporal data
            temporal_anchor_vector = torch.tensor(0.5).view(1, 1, 1, 1).float().to(device).repeat(current_frame_tensor.shape[0], 1, 1, 1)
            
            # D2. Predict Flow/Mask between neighbors
            flow_prev, flow_next, merge_mask = flow_prediction_engine(neighbor_frame_prev, neighbor_frame_next, temporal_anchor_vector, onlyFlow=True)

            # D3. Predict Features for neighbors
            features_prev = feature_extractor(neighbor_frame_prev)
            features_next = feature_extractor(neighbor_frame_next)
            
            # D4. Fuse Features
            feature_set = [features_prev, current_features, features_next]
            flow_set = [flow_prev, flow_next]
            fused_features = temporal_fusion_unit(feature_set, flow_set, merge_mask)
            
            # D5. Predict Fused Disparity
            fused_outputs = disparity_predictor_fused(fused_features) 
            fused_disparity = fused_outputs[("disp", 0)]
            fused_disparity_resized = F.interpolate(
                fused_disparity, (original_height, original_width), mode="bilinear", align_corners=False)
            
            
            # E. --- Saving Results (Mono) ---
            output_name = os.path.splitext(os.path.basename(current_path))[0]
            
            # E1. Compute/Save Numpy (Mono)
            if config.save_numpy_data:
                scaled_disp, depth = disp_to_depth(mono_disparity_resized, config.depth_min, config.depth_max)
                name_dest_npy = os.path.join(output_directory, f"{output_name}_mono_disp.npy")
                np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # E2. Generate Colormap (Mono)
            disp_np = mono_disparity_resized.squeeze().cpu().numpy()
            vmax_mono = np.percentile(disp_np, 95)
            normalizer_mono = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax_mono)
            mapper_mono = cm.ScalarMappable(norm=normalizer_mono, cmap='magma')
            colormapped_mono = (mapper_mono.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
            
            name_dest_im_mono = os.path.join(output_directory, f"{output_name}_mono_disp.jpeg")
            pil.fromarray(colormapped_mono).save(name_dest_im_mono)

            # F. --- Saving Results (Fused) ---

            # F1. Compute/Save Numpy (Fused)
            if config.save_numpy_data:
                scaled_disp_mf, depth_mf = disp_to_depth(fused_disparity_resized, config.depth_min, config.depth_max)
                name_dest_npy_mf = os.path.join(output_directory, f"{output_name}_fused_disp.npy")
                np.save(name_dest_npy_mf, scaled_disp_mf.cpu().numpy())

            # F2. Generate Colormap (Fused)
            disp_mf_np = fused_disparity_resized.squeeze().cpu().numpy()
            vmax_fused = np.percentile(disp_mf_np, 95)
            normalizer_fused = mpl.colors.Normalize(vmin=disp_mf_np.min(), vmax=vmax_fused)
            mapper_fused = cm.ScalarMappable(norm=normalizer_fused, cmap='magma')
            colormapped_fused = (mapper_fused.to_rgba(disp_mf_np)[:, :, :3] * 255).astype(np.uint8)
            
            name_dest_im_fused = os.path.join(output_directory, f"{output_name}_fused_disp.jpeg")
            pil.fromarray(colormapped_fused).save(name_dest_im_fused)

            # G. --- Prepare GIF Frame ---
            # Concatenate original image, mono disp, and fused disp vertically
            combined_image_np = np.concatenate([original_image_np, colormapped_mono, colormapped_fused], axis=0)
            
            combined_image_pil = pil.fromarray(combined_image_np)
            # Resize for the final GIF (downscaling by factor of 2)
            combined_image_pil = combined_image_pil.resize(
                (feed_width // 2, feed_height // 2 * 3), pil.LANCZOS
            )
            gif_frames_output.append(combined_image_pil)
        
        # 5. Save GIF Output
        gif_output_name = "demo.gif"
        if gif_frames_output:
            gif_frames_output[0].save(
                gif_output_name, save_all=True, append_images=gif_frames_output[1:], 
                duration=150, loop=0
            )
            print(f" Save output gif to: {gif_output_name}")

    print('-> Done!')


if __name__ == '__main__':
    # Add external dependencies for plotting before execution
    import matplotlib
    matplotlib.use('Agg')
    
    test_config = parse_test_configuration()
    run_video_pipeline(test_config)
