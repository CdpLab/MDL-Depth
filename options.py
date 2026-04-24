import os
import configargparse

# Determine the script's root directory
source_location = os.path.dirname(__file__)

# Initialize the configuration parser for Deep Learning Model Deployment (MDL-Depth options in original)
config_handler = configargparse.ArgumentParser(description="Deep Learning Model Deployment Configuration")

# --- I. Core Setup ---
config_handler.add_argument('-c', '--setup-file', required=True, is_config_file=True, 
                            help='Path to the dedicated configuration file.')

# --- II. Distributed Training Configuration ---
config_handler.add_argument('--local_rank', 
                            default=0, 
                            type=int,
                            help='Node rank within the current machine for distributed processing.')
config_handler.add_argument('--global_rank', 
                            default=0, 
                            type=int,
                            help='Global node rank across all machines.')
config_handler.add_argument('--world_size', 
                            default=1, 
                            type=int,
                            help='Total count of processing units (GPUs/nodes) used.')

# --- III. Resource Locations ---
config_handler.add_argument("--primary_data_source",
                            type=str,
                            help="Root path to the primary training dataset.",
                            default=os.path.join(source_location, "kitti_data"))
config_handler.add_argument("--preprocessed_data_source",
                            type=str,
                            help="Path to preprocessed data (e.g., for Cityscapes).")
config_handler.add_argument("--output_repository",
                            type=str,
                            help="Directory where logs and checkpoints will be stored.",
                            default=os.path.join(os.path.expanduser("~"), "tmp"))

# --- IV. Model Training Parameters ---
config_handler.add_argument("--project_identifier",
                            type=str,
                            help="A unique name for the experiment and output folder.",
                            default="mdp")
config_handler.add_argument("--data_partition",
                            type=str,
                            help="Which data subset definition to use for training (e.g., 'eigen_zhou').",
                            choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                            default="eigen_zhou")
config_handler.add_argument("--validation_partition",
                            type=str,
                            default="eigen",
                            choices=["eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                            help="Which data subset to use for quantitative evaluation.")
config_handler.add_argument("--encoder_depth",
                            type=int,
                            help="Number of layers in the feature extraction backbone (e.g., 18, 50).",
                            default=18,
                            choices=[18, 34, 50, 101, 152])
config_handler.add_argument("--target_collection",
                            type=str,
                            help="The name of the dataset being used for training.",
                            default="kitti",
                            choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", "nyuv2", "cityscapes"])
config_handler.add_argument("--use_jpeg_format",
                            help="Flag to use JPEG files instead of default PNGs.",
                            action="store_true")
config_handler.add_argument("--input_resolution_v",
                            type=int,
                            help="Input image height (vertical resolution).",
                            default=192)
config_handler.add_argument("--input_resolution_h",
                            type=int,
                            help="Input image width (horizontal resolution).",
                            default=640)
config_handler.add_argument("--inverse_depth_regularizer",
                            type=float,
                            help="Weighting factor for the predicted depth smoothness constraint.",
                            default=1e-3)
config_handler.add_argument("--multi_scale_levels",
                            type=int,
                            help="Number of multi-scale levels utilized in the reconstruction loss.",
                            default=1)
config_handler.add_argument("--depth_range_min",
                            type=float,
                            help="Minimum valid depth value in meters.",
                            default=0.1)
config_handler.add_argument("--depth_range_max",
                            type=float,
                            help="Maximum valid depth value in meters.",
                            default=100.0)
config_handler.add_argument("--depth_consistency_factor",
                            type=float,
                            help="Coefficient for the depth consistency/distillation loss term (lambda in original).",
                            default=0.2)
config_handler.add_argument("--enable_stereo_mode",
                            help="If set, training incorporates stereo image pairs.",
                            action="store_true")
config_handler.add_argument("--temporal_context_indices",
                            nargs="+",
                            type=int,
                            help="Temporal indices of frames to load relative to the target frame (e.g., [0, -1, 1]).",
                            default=[0, -1, 1])

# --- V. Learning Dynamics ---
config_handler.add_argument('--optimization_strategy',
                            type=str,
                            default='adamw',
                            help='Optimization algorithm choice.',
                            choices=["adamw", "adam", "sgd"])
config_handler.add_argument('--rate_schedule_profile',
                            type=str,
                            default='step',
                            help='Learning rate scheduler type.',
                            choices=["cos", "step"])
config_handler.add_argument('--minimum_rate_bound',
                            type=float,
                            default=5e-6,
                            help='The minimum final learning rate when using cosine annealing.')
config_handler.add_argument("--data_batch_size",
                            type=int,
                            help="Number of samples processed per iteration.",
                            default=12)
config_handler.add_argument('--initial_rate',
                            default=0.0001,
                            type=float,
                            help='Starting value for the learning rate.')
config_handler.add_argument('--rate_drop_multiplier',
                            dest='rate_drop_multiplier',
                            type=float,
                            default=0.1,
                            help='Factor by which the learning rate is multiplied during decay.')
config_handler.add_argument('--drop_epochs',
                            dest='drop_epochs',
                            type=int,
                            nargs='+',
                            default=[15],
                            help='Epoch numbers at which the learning rate decay is applied.')
config_handler.add_argument('--L2_regularization',
                            type=float,
                            default=0.01,
                            help='Weight decay coefficient for AdamW.')
config_handler.add_argument('--momentum_factor_1',
                            type=float,
                            default=0.9,
                            help='Beta1 parameter for Adam/AdamW.')
config_handler.add_argument('--momentum_factor_2',
                            type=float,
                            default=0.999,
                            help='Beta2 parameter for Adam/AdamW.')
config_handler.add_argument('--SGD_velocity',
                            default=0.9,
                            type=float,
                            help='Momentum parameter for SGD optimizer.')
config_handler.add_argument('--gradient_clip_threshold',
                            dest='gradient_clip_threshold',
                            type=float,
                            default=5,
                            help='Maximum L2 norm for gradient clipping.')
config_handler.add_argument("--total_training_epochs",
                            type=int,
                            help="Total number of complete training cycles.",
                            default=20)
config_handler.add_argument("--random_initialization_seed",
                            type=int,
                            help="Seed value for all random processes.",
                            default=1234)
config_handler.add_argument("--continue_from_checkpoint",
                            help="If set, attempts to load the last saved checkpoint and resume training.",
                            action="store_true")

# --- VI. Experimental Controls ---
config_handler.add_argument("--average_photometric_loss",
                            help="If set, computes the average reprojection loss instead of the minimum.",
                            action="store_true")
config_handler.add_argument("--bypass_auto_masking",
                            help="Disables the self-masking mechanism for static pixel filtering.",
                            action="store_true")
config_handler.add_argument("--exclude_ssim_metric",
                            help="Removes the Structural Similarity Index from the photometric loss function.",
                            action="store_true")
config_handler.add_argument("--network_initialization_source",
                            type=str,
                            help="Source for initial network weights.",
                            default="pretrained",
                            choices=["pretrained", "scratch"])
config_handler.add_argument("--feature_extractor_architecture",
                            type=str,
                            default="ResNet18",
                            choices=["ResNet18", "ResNet50", "LSM"])
config_handler.add_argument("--flow_network_complexity",
                            type=str,
                            help="Defines the complexity scale of the GMNet/flow network.",
                            default="small",
                            choices=["large", "small"])
config_handler.add_argument("--fusion_module_configuration",
                            type=str,
                            help="Configuration setting for sharing encoders/decoders between branches.",
                            default="shared_encoder",
                            choices=["shared_encoder", "separate_all", "shared_all"])
config_handler.add_argument("--apply_affine_transform",
                            help="Enables affine augmentation during data loading.",
                            action="store_true")

# --- VII. Operational Settings ---
config_handler.add_argument("--data_loader_threads",
                            type=int,
                            help="Number of parallel workers for data loading.",
                            default=16)

# Loading options
config_handler.add_argument("--model_load_path",
                            type=str,
                            help="File path to an existing trained model for fine-tuning or testing.")

# Logging options
config_handler.add_argument("--logging_interval_batches",
                            type=int,
                            help="Number of batches between each console logging update.",
                            default=500)
config_handler.add_argument("--checkpoint_interval_epochs",
                            type=int,
                            help="Number of epochs between saving model checkpoints.",
                            default=500)

settings = config_handler.parse_args()
