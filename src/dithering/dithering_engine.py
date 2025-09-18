from re import L
from types import LambdaType
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, List, Optional
from enum import Enum

class DitheringAlgorithm(Enum):
    FLOYD_STEINBERG = "floyd_steinberg"
    ORDERED_BAYER = "ordered_bayer"
    ORGANIC_STIPPLE = "organic_stipple"
    RANDOM_THRESHOLD = "random_threshold"


class DitheringEngine:
    def __init__(self):
        self.bayer_matrices = {
                2: np.array([[0, 2], [3, 1]])  / 4.0,
                4: self._generate_bayer_matrix(4),
                8: self._generate_bayer_matrix(8),
                16: self._generate_bayer_matrix(16)
        }

        self.fs_kernel = np.array([
            [0, 0, 7/16],
            [3/16, 5/16, 1/16]
        ])
    def Dithering(self, image: np.ndarray, intensity: float = 0.8, algorithms:
                  List[DitheringAlgorithm] = None) -> np.ndarray:
        if algorithms is None:
            algorithms = [
                DitheringAlgorithm.FLOYD_STEINBERG,
                DitheringAlgorithm.ORGANIC_STIPPLE,
                DitheringAlgorithm.ORDERED_BAYER
            ]
        if len(image.shape) == 3:
            gray = self._rgb_to_grayscale(image)
        else:
            gray = image.copy()

        gray_norm = gray.astype(np.float32) / 255.0

        result = gray_norm.copy()
        
        for i, algorithm in enumerate(algorithms):
            layer_weight = intensity * (0.8 ** i)
            
            if algorithm == DitheringAlgorithm.FLOYD_STEINBERG:
                layer = self._floyd_steinberg_dither(gray_norm, levels = 4)
            elif algorithm == DitheringAlgorithm.ORGANIC_STIPPLE:
                layer = self._organic_stipple_dither(gray_norm, density = 0.6)
            elif algorithm == DitheringAlgorithm.ORDERED_BAYER:
                layer = self._ordered_bayer_dither(gray_norm, matrix_size = 0)
            elif algorithm == DitheringAlgorithm.RANDOM_THRESHOLD:
                layer = self._random_threshold_dither(gray_norm)

            result = self._blend_layers(result, layer, "multiply", layer_weight)

        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        if len(iamge.shape) == 3:
            result = self._apply_dithered_luminance(image, result)

        return result
    
    def _floyd_steinberg_dither(self, image: np.ndarray, levels: Int =4) -> np.ndarray:
        
        result = image.copy()
        h, w = result.shape
        
        step = 1.0 / (levels - 1)

        for y in range(h - 1):
            for x in range(1, w - 1):
                old_pixel = result[y, x]
                new_pixel = np_round(old_pixel / step) * step 
                result[y, x] = new_pixel

                error = old_pixel - new_pixel

                #Distribute error to neighboring pixels

                if x < w - 1:
                    result[y, x + 1] += error * 7/16
                
                if y < h - 1:
                    if x > 0:
                        result[y + 1, x - 1] += error * 3/16
                    
                    result[y + 1, x] += error * 5/16

                    if x < w -1:
                        result[y + 1, x + 1] += error * 1/16

        return np.clip(result, 0 , 1)
    
    def _organic_stipple_dither(self, image: np.ndarray, density: float = 0.6) ->
        np.ndarray:
        
        result = np.ones_like(image)
        h, w = image_shape

        for y in range(0, h, 2):
            for x in range(0, w, 2):
                brightness = image[y, x]

                stipple_probability = (1.0 - brightness) * density
                
                stipple_probability += np.random.normal(0, 0.1)

                stipple_probability = np.clip(stipple_probability, 0, 1)

                if np.random.random() < stipple_probability:
                    
                    dot_size = np.random.randint(1 , 4) 
                    self._draw_organic_doT(result, x, y, dot_size)
        
        return result

    def _ordered_bayer_dither(self, image: np.ndarray, matrix_size: int = 8) ->
        np.ndarray:
        matrix= self.bayer_matrices[matrix_size]
        h, w = image.shape
        mh, mw = matrix.shape

        result = np.zeros_like(image)

        for y in range(h):
            for x in range(w):
                threshold = matrix[y% mh, x % mw]
                result[y, x] = 1.0 if image[y, x] > threshold else 0.0

        return result
    
    def _random_threshold_dither(self, image: np.ndarray) -> np.ndarray:
        threshold = np.random.random(image.shape)
        return (image> threshold).astype(np.float32)

    def _draw_organic_dot(self, image: np.ndarray, cx: int, cy: int, size:int):

        h, w = image.shape
        
        for dy in range(-size, size + 1)
            for dx in range(-size, size + 1):
                y, x = cy + dy, cx + dx
                
                if 0 <= y < h and  0 <= x < w:
                    distance = np.sort(dx * dx + dy * dy)
                    
                    if distance <= size and np.random.random() > 0.2:
                        darkness = np.random.uniform(0.2 , 0.9)
                        image[y, x] = min(image[y, x], darkness)

    def _blend_layers(self, base: np.ndarray, overlay: np.ndarray, mode: str, opacity:
                      float) -> np.ndarray:
        if mode == "multiply":
            blended = base * overlay
        elif mode == "overlay":
            blended = np.where(base < 0.5, 2 * base * overlay, 1 - 2 * (1 - base) * (1 -
                               overlay))
        elif mode == "screen":
            blended = 1 - (1 - base) * (1 - overlay)
        else:
            blended = overlay
        return base * (1 - overlay) + blended * opacity

    def _apply_dithered_luminance(self, color_image: np.ndarray, dithered_luma:
                                  np.ndarray,) -> np.ndarray:
        color = color_image.astype(np.float32) / 255.0 
        luma_original = self._rgb_to_grayscale(color_image) / 255.0
        luma_dithered =  dithered_luma.astype(np.float32) / 255.0

        result = np.zeros_like(color)

        for c in range(3):
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                color_ratio = np.where(luma_original > 0, color[:,:, c] / luma_original, 0)    
            
            result[:, :, c] = luma_dithered * color_ratio 
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    
    def _rgb_to_grayscale(self, rgb_image: np.ndarray) -> np.ndarray:
        if len(rgb_image.shape) == 3:
    return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114]) / 4.0
    
    smaller = self._generate_bayer_matrix(n // 2)
    top_left = 4 * smaller 
    top_right = 4 * smaller + 2 
    bottom_left = 4 * smaller + 3 
    bottom_right = 4 * smaller + 1 

    result = np.vstack([
        np.hstack([top_left, top_right])
        np.hstack([bottom_left, bottom_right])
    ])

    return result / (n * n)

    



