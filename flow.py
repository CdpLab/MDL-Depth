import numpy as np

class FlowColorMapper:
    """
    Encapsulates logic for converting 2D optical flow vectors into the standard 
    color visualization scheme (based on the Middlebury flow benchmark).
    """

    def __init__(self):
        # Define the fixed lengths for the six color segments (R-Y-G-C-B-M)
        self.segment_lengths = {
            "R_Y": 15,  # Red to Yellow
            "Y_G": 6,   # Yellow to Green
            "G_C": 4,   # Green to Cyan
            "C_B": 11,  # Cyan to Blue
            "B_M": 13,  # Blue to Magenta
            "M_R": 6    # Magenta to Red
        }
        self.flow_color_palette = self._initialize_palette_segments()
        self.total_palette_size = self.flow_color_palette.shape[0]

    def _initialize_palette_segments(self):
        """
        Generates a color palette array (color wheel) by interpolating between 
        the primary and secondary colors. Equivalent to the original make_colorwheel function.
        """
        L = self.segment_lengths
        num_cols = sum(L.values())
        palette = np.zeros((num_cols, 3))
        c = 0 # Current column index

        # R -> Y
        palette[c:c + L["R_Y"], 0] = 255
        palette[c:c + L["R_Y"], 1] = np.floor(255 * np.arange(L["R_Y"]) / L["R_Y"])
        c += L["R_Y"]
        
        # Y -> G
        palette[c:c + L["Y_G"], 0] = 255 - np.floor(255 * np.arange(L["Y_G"]) / L["Y_G"])
        palette[c:c + L["Y_G"], 1] = 255
        c += L["Y_G"]
        
        # G -> C
        palette[c:c + L["G_C"], 1] = 255
        palette[c:c + L["G_C"], 2] = np.floor(255 * np.arange(L["G_C"]) / L["G_C"])
        c += L["G_C"]
        
        # C -> B
        palette[c:c + L["C_B"], 1] = 255 - np.floor(255 * np.arange(L["C_B"]) / L["C_B"])
        palette[c:c + L["C_B"], 2] = 255
        c += L["C_B"]
        
        # B -> M
        palette[c:c + L["B_M"], 2] = 255
        palette[c:c + L["B_M"], 0] = np.floor(255 * np.arange(L["B_M"]) / L["B_M"])
        c += L["B_M"]
        
        # M -> R
        palette[c:c + L["M_R"], 2] = 255 - np.floor(255 * np.arange(L["M_R"]) / L["M_R"])
        palette[c:c + L["M_R"], 0] = 255
        
        return palette

    def _map_normalized_vectors_to_rgb(self, vector_x, vector_y, output_bgr=False):
        """
        Maps normalized flow vectors (x, y) to a color visualization image.
        Equivalent to the original flow_uv_to_colors function.
        
        Args:
            vector_x (np.ndarray): Normalized horizontal flow component.
            vector_y (np.ndarray): Normalized vertical flow component.
            output_bgr (bool): If True, returns BGR instead of RGB.
        """
        viz_image = np.zeros((vector_x.shape[0], vector_x.shape[1], 3), np.uint8)
        
        palette = self.flow_color_palette
        num_cols = self.total_palette_size
        
        # 1. Calculate magnitude (radius) and angle
        magnitude = np.sqrt(np.square(vector_x) + np.square(vector_y))
        
        # Angle mapping: a = arctan2(-v, -u) / pi. Range [-1, 1]
        angle_norm = np.arctan2(-vector_y, -vector_x) / np.pi
        
        # 2. Calculate floating point index and integer indices for lookup
        float_index = (angle_norm + 1) / 2 * (num_cols - 1)
        
        idx_int_low = np.floor(float_index).astype(np.int32)
        idx_int_high = idx_int_low + 1
        
        # Wrap the high index around 
        idx_int_high[idx_int_high == num_cols] = 0
        
        interpolation_factor = float_index - idx_int_low
        
        # 3. Interpolate colors and apply saturation
        for i in range(palette.shape[1]):
            channel_colors = palette[:, i]
            
            c_low = channel_colors[idx_int_low] / 255.0
            c_high = channel_colors[idx_int_high] / 255.0
            
            # Linear blend
            blended_color = (1 - interpolation_factor) * c_low + interpolation_factor * c_high
            
            # Saturation: vectors inside unit circle (mag <= 1) are saturated
            inside_circle_mask = (magnitude <= 1)
            blended_color[inside_circle_mask] = 1 - magnitude[inside_circle_mask] * (1 - blended_color[inside_circle_mask])
            
            # Desaturation: vectors outside unit circle (mag > 1) are dimmed
            blended_color[~inside_circle_mask] = blended_color[~inside_circle_mask] * 0.75
            
            # Set the final color channel (2-i for BGR, i for RGB)
            channel_idx = 2 - i if output_bgr else i
            viz_image[:, :, channel_idx] = np.floor(255 * blended_color)
            
        return viz_image

    def convert_raw_flow_to_visualization(self, raw_flow_vectors, flow_magnitude_limit=None, output_bgr=False):
        """
        Primary entry point. Normalizes raw flow vectors and generates the color visualization.
        Equivalent to the original flow_to_color function.
        
        Args:
            raw_flow_vectors (np.ndarray): Flow UV image of shape [H,W,2].
            flow_magnitude_limit (float, optional): Clip maximum of flow values.
            output_bgr (bool): Convert output image to BGR.
        """
        if raw_flow_vectors.ndim != 3 or raw_flow_vectors.shape[2] != 2:
            raise ValueError('Input flow vectors must have shape [H, W, 2]')
            
        if flow_magnitude_limit is not None:
            # Clip the raw flow values
            raw_flow_vectors = np.clip(raw_flow_vectors, 0, flow_magnitude_limit)
            
        x_comp = raw_flow_vectors[:, :, 0]
        y_comp = raw_flow_vectors[:, :, 1]
        
        # Calculate max magnitude for normalization
        magnitude = np.sqrt(np.square(x_comp) + np.square(y_comp))
        max_magnitude = np.max(magnitude)
        
        # Normalize vectors to unit circle
        epsilon = 1e-5
        normalized_x = x_comp / (max_magnitude + epsilon)
        normalized_y = y_comp / (max_magnitude + epsilon)
        
        return self._map_normalized_vectors_to_rgb(normalized_x, normalized_y, output_bgr)
