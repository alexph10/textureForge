from ctypes import DllCanUnloadNow
import numpy as np 
from PIL import Image 
import cv2
from typing import Tuple, List, Optional, Literal
from sklearn.cluster import KMeans

class ColorQuantizer:
    def __init__:
        self.refined_palettes = {
                "palette_1": np.array([
                    [28,23, 28],
                    [39, 41, 32],
                    [68, 58, 44],
                    [85, 77, 56],
                    [76, 143, 125]
                ]),
                "palette_2": np.array([
                    [19, 19, 19],
                    [39, 41, 32],
                    [76, 59, 48],
                    [188, 83, 66],
                    [173, 68, 44]
                ]),
                "palette_3": np.array([
                    [31, 22, 15],
                    [74, 50, 30],
                    [109, 94, 54],
                    [204, 193, 181],
                    [153, 50, 30]
                ]),
                "palette_4": np.array([
                    [26, 25, 24],
                    [63, 60, 56],
                    [110, 101, 93],
                    [284, 193, 181],
                    [153, 50, 30]
                ])
        }
        
        # Parameters based on the reference images

        self.target_wamth = 65.8
        self.shadow_dominance_min = 0.35
        self.accent_coverage_range = (0.01, 0.18)
        
        # Saturation stratification targets
        
        self.base_saturation_range = (0.2, 0.4)
        self.accent_saturation_range = (0.5, 0.8)

        # Brightness distribution targets

        self.target_distribution = {
                "shadows": 0.40,
                "midtones": 0.45,
                "highlights": 0.153
        }
    def shadow_dominance(self, image: np.ndarray, min_shadow_percentage: float = 0.35) -> np.ndarray:
        
        #Converting image to working format
    
        img_float = image.astype(np.float32) / 255.0
        
        #Calculating brightness for each pixel

        brightness = np.sum(img_float, axis = 2) / 3.0
        
        # Count current shadow pixels (brightness < 0.31 = 80/255)

        shadow_mask = brightness < 0.31
        current_shadow_percentage = np.sum(shadow_mask) / shadow_mask.size

        if current_shadow_percentage < min_shadow_percentage:
            shortage = min_shadow_percentage - current_shadow_percentage
            pixels_to_darken = int(shortage * shadow_mask.size)
            
            #Find pixels that are close to shadow range (0.31 - 0.5 brightness)
            near_shadow_mask = (brightness >= 0.31) & (brightness < 0.5)
            near_shadow_indices = np.where(near_shadow_mask)

            if len(near_shadow_indices[0]) > 0: 
                #Random pixels selected to darken
                selection_indices = np.random.choice(len(near_shadow_indices[0]), min(pixels_to_darken, len(near_shadow_indices[0])), replace= False)
                selected_y = near_shadow_indices[0][selection_indices]
                selected_x = near_shadow_indices[1][selection_indices]

                # Darken selected pixels by 40-60%
                
                darkening_factor = np.random.uniform(0.4, 0.6, len(selected_y))
                
                for i, (y, x) in enumerate(zip(selected_y, selected_x)):
                    img_float[y, x] *= darkening_factor[i]

        return (img_float * 255).astype(np.uint8)
    
    def apply_selective_accent_placement(self, image: np.ndarray, accent_color: Tuple[int, int, int], target_coverage: float = 0.05) -> np.ndarray:

        img_float = image.astype(np.float32) / 255.0

        accent_float = np.array(accent_color, dtype = np.float32) / 255.0

        saliency_map = self._calculate_visual_saliency(img_float)

        total_pixels = img_float.shape[0] * img_float.shape[1]
        target_pixels = int(total_pixels * target_coverage)

        flat_saliency = saliency_map.flatten()
        threshold_idx = len(flat_saliency) - target_pixels
        saliency_threshold = np.partition(flat_saliency, threshold_idx)[threshold_idx]

        accent_mask = saliency_map >= saliemcy_threshold

        result = img_float.copy()
        result[accent_mask] = accent_float

        return (result * 255).astype(np.uint8)


    def _calculate_visual_saliency(self, image: np.ndarray) -> np.ndarray:
        
        # Convert  to grayscale for edge detection 
        
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        gray_uint8 = (gray * 255).astype(np.uint8)

        #Calculate edge strength using Sobel operators

        grad_x = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)

        edge_strength = np.sqrt(grad_x ** 2 + grad_y ** 2)

        h, w = gray.shape
        y_center, x_center = h // 2, w // 2 
        y_coords, x_coords = np.ogrid[:h, :w]
        center_distance = np.sqrt((y_coords - y_center) ** 2 + (x_coords - x_center) ** 2)
        max_distance = np.sqrt(y_center ** 2 + x_center ** 2)
        center_bias = 1.0 - (center_distance / max_distance)

        # Calculate contrast (local standard deviation)
        kernel = np.ones((5, 5), np.float32) / 25 
        local_mean = cv2.filter2D(gray, -1, kernel)
        local_variance = cv2.filter2D(gray ** 2, -1, kernel) - local_mean ** 2 
        contrast = np.sqrt(np.maximum(local_variance, 0))

        # Combine factors for final saliency 

        saliency = (
            0.4 * (edge_strength / edge_strength.max()) +
            0.3 * center_bias +
            0.3 * (contrast/ contrast.max())
        )
        
        return saliency
    
    def apply_saturation_stratification(self, image: np.ndarray, palette: np.ndarray, is_accent_mask: np.ndarray) -> np.ndarray:
        modified_palette = palette.copy().astype(np.float32)

        for i, color in enumerate (modified_palette):

            color_rgb = color.reshape(1, 1, 3).astype(np.uint8)
            color_hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            h, s, v = color_hsv[0, 0, 0], color_hsv[0, 0, 1], color_hsv[0, 0, 2]

            # Determine if this is an accent color

            # (Simplified - in practice you'd use more sophisticated detection)
            current_saturation = s / 255.0
            if current_saturation > 0.5:
                target_sat = np.clip(current_saturation, self.accent_saturation_range[0], self.base_saturation_range[1])
            else:
                target_sat = np.clip(current_saturation, self.base_saturation_range[0], self.base_saturation_range[1])

            color_hsv[0, 0, 1] = target_sat * 255
            modified_rgb = cv2.cvtColor(color_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            modified_palette[i] = modified_rgb[0, 0].astype(np.float32)
        return modified_palette.astype(np.uint8)
    
    def enforce_brightness_distribution(self, image: np.ndarray) -> np.ndarray:
        img_float = image.astype(np.float32) / 255.0 
        
        brightness = np.sum(img_float, axis = 2) / 3.0
        
        shadow_mask = brightness < (80/255)
        
        highlight_mask = brightness > (150/255)

        midtone_mask  = ~(shadow_mask | highlight_mask)

        total_pixels = brightness.size

        current_shadows = np.sum(shadow_mask) / total_pixels
        current_highlights = np.sum(highlight_mask) / total_pixels
        current_midtones = np.sum(midtone_mask) / total_pixels

        target_shadows = self.target_distribution['shadows']
        target_highlights = self.target_distribution['highlights']

        result = img_float.copy()

        if current_shadows < target_shadows:
            shortage = target_shadows - current_shadows
            pixels_to_darken = int(shortage * total_pixels)
            
            midtone_brightness = brightness.copy()
            midtone_brightness[~midtone_mask] = -1

            flat_brightness = midtone_brightness.flatten()
            valid_indices = np.where(flat_brightness > 0)[0]
            
            if len(valid_indices) > 0:
                n_to_select = min(pixels_to_darken, len(valid_indices))
                brightest_indices = np.argpartition(
                    flat_brightness[valid_indices], - n_to_select
                )[-n_to_select:]

                selected_global_indices = valid_indices[brightest_indices]
                y_coords = selected_global_indices // result.shape[1]
                x_coords = selected_global_indices %  result.shape[1] 

                for y,x in zip(y_coords, x_coords):
                    result.[y, x] *= 0.6

        if current_highlights > target_highlights:
            excess = current_highlights - target_highlights
            pixels_to_tone = int(excess * total_pixels)
            
            highlight_indices = np.where(highlight_mask.flatten())[0]

            if len(highlight_indices) > pixels_to_tone:
                selected_indices = np.random.choice(
                    highlight_indices, pixels_to_tone, replace=False
                )

                y_coords = selected_indices // result.shape[1]
                x_coords = selected_indices % result.shape[1]
                
                for y,x in zip(y_coords, x_coords):
                    result[y,x] *= 0.8 
        return (result* 255).astype(np.uint8)
    
    def refined_quantization(self, image: np.ndarray, target_colors: int = 4, palette_style: str = "auto", accent_coverage: float = 0.05) -> np.ndarray:
        distributed = self.enforce_brightness_distribution(image)
        shadow_enforced = self.ensure_shadow_dominance(distributed, self.shadow_dominance_min)

        if palette_style == "auto":
            avg_brightness = np.mean(shadow_enforced) / 255.0 
            palette_style = "palette_1" if avg_brightness < 0.4 else "palette_2"
        
        img_float = shadow_enforced.astype(np.float32) / 255.0
        pixels = img_float.reshape(-1, 3)

        kmeans = KMeans(n_clusters=target_colors, random_state = 42, n_init = 10)

        kmeans.fit(pixels)

        quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
        
        quantized = quantized_pixels.reshape(img_float.shape)
        
        quantized_uint8 = (quantized * 255).astype(np.uint8)

        palette = kmeans.cluster_centers_

        accent_idx = self._identify_accent_color(palette)
        if accent_idx is not None:
            accent_color = (palette[accent_idx] * 255).astype(int)
            quantized_uint8 = self.apply_selective_accent_placement(
                quantized_uint8, tuple(accent_color), accent_coverage)

        return quantized_uint8
    
    def _identify_accent_color(self, palette: np.ndarray) -> Optional[int]:
        
        if len(palette) < 3:
            return None
        saturations = []
        for color in palette:
            max_val = np.max(color)
            min_val = np.min(color)
            sat = (max_val - min_val) / max(max_val, 0.001)
            saturations.append(sat)

        saturations = np.array(saturations)
        
        if np.max(saturations) > 0.3:
            return np.argmax(saturations)

        return None
        



