from PIL import Image
import torch
import matplotlib.pyplot as plt

class ImageUtils():
    # -- Image Tiling ---
    """Utility functions for image processing."""
    @staticmethod
    def get_image_tiles(image_tensor, tile_size=20, stride=10, return_indices=False):
        c, h, w = image_tensor.shape
        tiles = image_tensor.unfold(1, tile_size, stride).unfold(2, tile_size, stride)
        h_tiles, w_tiles = tiles.shape[1], tiles.shape[2]
        tiles = tiles.contiguous().view(c, -1, tile_size, tile_size)
        tiles = tiles.permute(1, 0, 2, 3)
        flat_tiles = tiles.contiguous().view(tiles.size(0), -1)
        if return_indices:
            return flat_tiles, h_tiles, w_tiles
        return flat_tiles
    # --- Image Tiling Debugging Function ---
    def debug_image_tiling(self, original_image_tensor, tile_size=20, stride=10):
        """
        Visually debugs the get_image_tiles function by reconstructing the image from its tiles.
        
        Args:
            original_image_tensor (torch.Tensor): The input image tensor [C, H, W].
            tile_size (int): The size of the tiles.
            stride (int): The step size between tiles.
        """
        print("\n--- STARTING TILING DEBUGGER ---")
        # 1. Get the tiles from the image (now returns CPU tensor)
        tiles = self.get_image_tiles(original_image_tensor, tile_size, stride)
        
        # 2. Reshape tiles back to image format
        c, h, w = original_image_tensor.shape
        num_channels = c
        reshaped_tiles = tiles.view(-1, num_channels, tile_size, tile_size)
        
        # Calculate number of tiles in each dimension
        num_tiles_h = (h - tile_size) // stride + 1
        num_tiles_w = (w - tile_size) // stride + 1

        # 3. Prepare a blank canvas for reconstruction
        reconstructed_image = torch.zeros(3, num_tiles_h * tile_size, num_tiles_w * tile_size)
        
        print(f"Reconstructing image from {num_tiles_h} x {num_tiles_w} = {reshaped_tiles.shape[0]} tiles.")
        print(f"Original image shape:", original_image_tensor.size())
        print(f"Reconstructed image shape:", reconstructed_image.size())
        
        # 4. Place each tile onto the canvas
        tile_idx = 0
        for i in range(num_tiles_h):
            for j in range(num_tiles_w):
                # Calculate the position to place the tile
                y_pos = i * tile_size
                x_pos = j * tile_size
                
                # Place the tile
                reconstructed_image[:, y_pos:y_pos+tile_size, x_pos:x_pos+tile_size] = reshaped_tiles[tile_idx]
                tile_idx += 1
                
        # 5. Display the original and reconstructed images using Matplotlib
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Matplotlib expects image shape [H, W, C], so we permute the dimensions
        axes[0].imshow(original_image_tensor.permute(1, 2, 0))
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(reconstructed_image.permute(1, 2, 0))
        axes[1].set_title("Reconstructed from Tiles")
        axes[1].axis('off')
        
        plt.suptitle("Tiling Function Debugger", fontsize=16)
        plt.show()
        print("--- TILING DEBUGGER CLOSED ---")
