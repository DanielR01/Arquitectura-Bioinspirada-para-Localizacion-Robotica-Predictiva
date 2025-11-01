import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import torchvision.transforms as T
from tqdm import tqdm
import math
from enum import Enum
from torch.cuda.amp import autocast, GradScaler

class FiringSelectionMethod(Enum):
    WinnerTakeAll = 1
    StdFiringThreshold = 2
class PRM_L1_Visual_Debugger:
    def __init__(self, model, dataset, device, config):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.tile_size = config['tile_params']['size']
        self.stride = config['tile_params']['stride']
        self.init_tile = 0
        self.firing_std_factor = config['firing_std_factor']
        self.firing_selection_method = config['l1']['firing_selection_method']

        self.recon_data = {}
        
        self.fig = plt.figure(figsize=(18, 10))
        gs = self.fig.add_gridspec(2, 3)
        self.ax_image = self.fig.add_subplot(gs[0, 0]) #Shows the whole image that is being processed
        self.ax_tile = self.fig.add_subplot(gs[1, 0]) #Shows the selected tile to process
        self.ax_recon = self.fig.add_subplot(gs[1, 1]) #Shows the reconstructed pattern from the selected tile
        self.ax_info = self.fig.add_subplot(gs[1, 2])
        self.ax_heatmap = self.fig.add_subplot(gs[0, 1:]) #Heatmap of the prsms activation rates
        
        self.fig.suptitle('PRM Layer 1 Visual Debugger', fontsize=16)

        self.ax_image.set_title("Original Image")
        self.ax_image.axis('off')
        self.ax_tile.set_title("Selected Tile (Input)")
        self.ax_tile.axis('off')
        self.ax_recon.set_title("Reconstructed by Best PRsM (Click for details)")
        self.ax_recon.axis('off')
        self.ax_info.axis('off')
        self.ax_heatmap.set_title("PRsM Activation Frequency for this Image")

        plt.subplots_adjust(bottom=0.2)
        ax_img_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        ax_tile_slider = plt.axes([0.25, 0.05, 0.65, 0.03])
        ax_button_full = plt.axes([0.05, 0.1, 0.15, 0.04])
        ax_button_tile = plt.axes([0.05, 0.05, 0.15, 0.04])

        self.img_slider = Slider(ax_img_slider, 'Image Index', 0, len(self.dataset) - 1, valinit=0, valstep=1)
        self.tile_slider = Slider(ax_tile_slider, 'Tile Index', 0, 100, valinit=self.init_tile, valstep=1)

        self.run_tile_button = Button(ax_button_tile, 'Analyze Selected Tile')
        self.run_full_button = Button(ax_button_full, 'Analyze Full Image')

        self.img_slider.on_changed(self._update_display)
        self.tile_slider.on_changed(self._update_display)
        
        self.run_tile_button.on_clicked(self._perform_tile_forward_pass)
        self.run_full_button.on_clicked(self._perform_full_forward_pass)


        self.rect = patches.Rectangle((0, 0), self.tile_size, self.tile_size, linewidth=2, edgecolor='r', facecolor='none')
        self.ax_image.add_patch(self.rect)
        
        self.colorbar = None
        self.annot = self.ax_heatmap.annotate("", xy=(0,0), xytext=(20,20),
                                               textcoords="offset points",
                                               bbox=dict(boxstyle="round", fc="w"),
                                               arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)
        self.grid_shape = config['l1']['grid_shape']
        self.heatmap_data = np.zeros(self.grid_shape)

        self.cmap = plt.get_cmap('viridis').copy()
        self.cmap.set_under(color='black')
        self.extent = [-0.5, self.grid_shape[1] - 0.5, self.grid_shape[0] - 0.5, -0.5]
        self.heatmap_im = self.ax_heatmap.imshow(self.heatmap_data, cmap=self.cmap, aspect='auto', vmin=1, extent=self.extent)
        self.ax_heatmap.set_title("PRsM Activation Frequency for this Image (Black = 0 wins)")
        self.ax_heatmap.xaxis.set_major_locator(MultipleLocator(5))
        self.ax_heatmap.yaxis.set_major_locator(MultipleLocator(5))
        self.colorbar = self.fig.colorbar(self.heatmap_im, ax=self.ax_heatmap)
        
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.fig.canvas.mpl_connect('button_press_event', self._on_recon_click)
        self._update_display(0)
        plt.show()

    def _update_display(self, val):
        """Updates the display based on slider values without running the model."""
        img_idx = int(self.img_slider.val)
        
        dataset_item = self.dataset[img_idx]
        if dataset_item is None or dataset_item[0] is None:
            print(f"Warning: Could not load data for image index {img_idx}. Skipping.")
            return
            
        original_img, all_tiles, img_path = dataset_item
        
        if all_tiles is None: return
        
        num_tiles = all_tiles.shape[0]
        self.tile_slider.valmax = num_tiles - 1
        self.tile_slider.ax.set_xlim(0, num_tiles - 1)
        
        tile_idx = int(self.tile_slider.val)
        if tile_idx >= num_tiles:
            tile_idx = num_tiles - 1
            self.tile_slider.set_val(tile_idx)

        self.ax_image.imshow(original_img.permute(1, 2, 0))
        
        img_width = original_img.shape[2]
        num_tiles_w = (img_width - self.tile_size) // self.stride + 1
        row = tile_idx // num_tiles_w
        col = tile_idx % num_tiles_w
        self.rect.set_xy((col * self.stride, row * self.stride))
        
        selected_tile = all_tiles[tile_idx]

        self.ax_tile.imshow(selected_tile.cpu().view(3, self.tile_size, self.tile_size).permute(1, 2, 0))

        self.ax_recon.clear()
        self.ax_recon.set_title("Reconstructed by Best PRsM")
        self.ax_recon.axis('off')
        self.ax_info.clear()
        self.ax_info.axis('off')
        img_filename = os.path.basename(img_path)
        info_text = (f"Image: {img_filename}\n"
                     f"Tile: {tile_idx}\n\n"
                     f"Click 'Analyze' to see results.")
        self.ax_info.text(0, 0.5, info_text, fontsize=12, va='center')

        self.fig.canvas.draw_idle()

    def _apply_firing_threshold(self, errors):
        if self.firing_selection_method == FiringSelectionMethod.StdFiringThreshold:
            mean_errors = errors.mean(dim=1, keepdim=True)
            std_errors = errors.std(dim=1, keepdim=True)

            current_std_factor = self.firing_std_factor
            dynamic_threshold = mean_errors - (current_std_factor * std_errors)
            firing_mask = errors < dynamic_threshold

            # Iteratively lower threshold if no units fire
            while firing_mask.sum() == 0:
                current_std_factor -= 0.1
                dynamic_threshold = mean_errors - (current_std_factor * std_errors)
                firing_mask = errors < dynamic_threshold
                if current_std_factor < -5: # Safety break
                    break
            return firing_mask
        else: # WinnerTakeAll
            winner_indices = torch.argmin(errors, dim=1)
            firing_mask = torch.zeros_like(errors, dtype=torch.bool)
            firing_mask[torch.arange(errors.shape[0]), winner_indices] = True
            return firing_mask


    def _perform_tile_forward_pass(self, event):
        """Runs a fast forward pass on only the selected tile."""
        img_idx = int(self.img_slider.val)
        tile_idx = int(self.tile_slider.val)
        
        dataset_item = self.dataset[img_idx]
        if dataset_item is None or dataset_item[1] is None: return

        _, all_tiles, img_path = dataset_item
        
        selected_tile = all_tiles[tile_idx].unsqueeze(0).to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            recon, _ = self.model.layers['layer_1'](selected_tile)
            errors = F.mse_loss(recon, selected_tile.unsqueeze(1), reduction='none').mean(dim=2)
            print(f"Errors shape: {errors.shape}, Errors values: {errors}")
            
            firing_mask = self._apply_firing_threshold(errors)
            num_firing_prsms = firing_mask.sum().item()

            winner_idx = torch.argmin(errors, dim=1).item()
            best_recon = recon[0][winner_idx]
            recon_error = errors[0][winner_idx]

            self.recon_data['recon'] = recon
            self.recon_data['firing_mask'] = firing_mask

        self.ax_recon.imshow(best_recon.cpu().view(3, self.tile_size, self.tile_size).permute(1, 2, 0))
        
        img_filename = os.path.basename(img_path)
        info_text = (f"Image: {img_filename}\n"
                     f"Tile: {tile_idx}\n\n"
                     f"Firing PRsMs: {num_firing_prsms}\n"
                     f"Best PRsM: {winner_idx}\n"
                     f"Best Error: {recon_error:.10f}")
        self.ax_info.clear()
        self.ax_info.axis('off')
        self.ax_info.text(0, 0.5, info_text, fontsize=12, va='center')
        
        self.fig.canvas.draw_idle()

    def _perform_full_forward_pass(self, event):
        """Runs the forward pass on all tiles of the image and updates results."""
        img_idx = int(self.img_slider.val)
        
        dataset_item = self.dataset[img_idx]
        if dataset_item is None or dataset_item[1] is None: return

        _, all_tiles, _ = dataset_item
        
        all_tiles = all_tiles.to(self.device)
        self.model.eval()
        
        all_errors_list = []
        batch_size = 377
        with torch.no_grad():
            for tile_batch in tqdm(torch.split(all_tiles, batch_size)):
                recon, _ = self.model.layers['layer_1'](tile_batch)
                errors = F.mse_loss(recon, tile_batch.unsqueeze(1), reduction='none').mean(dim=2)
                all_errors_list.append(errors)
            
            all_errors = torch.cat(all_errors_list, dim=0)
            firing_mask = self._apply_firing_threshold(all_errors)
            win_counts = firing_mask.sum(dim=0).float()
            #print(f"\nTotal PRSMs active: {win_counts.sum().cpu()}")

        self.run_full_button.label.set_text('Analyze Full Image')
        
        self.heatmap_data = win_counts.reshape(self.grid_shape).cpu()
        
        self.heatmap_im.set_data(self.heatmap_data)
        
        max_val = self.heatmap_data.max()
        self.heatmap_im.set_clim(vmin=1, vmax=max_val if max_val > 0 else 1)
        
        if max_val > 0:
             tick_step = max(1, int(np.ceil(max_val / 10)))
             self.colorbar.set_ticks(np.arange(0, max_val + 1, tick_step))
        else:
             self.colorbar.set_ticks([])

        self.fig.canvas.draw_idle()

    def _on_hover(self, event):
        """Callback for mouse hover events on the heatmap."""
        if event.inaxes == self.ax_heatmap:
            col = int(round(event.xdata))
            row = int(round(event.ydata))

            if 0 <= row < self.grid_shape[0] and 0 <= col < self.grid_shape[1]:
                prsm_index = row * self.grid_shape[1] + col
                activation_count = self.heatmap_data[row, col]
                
                self.annot.xy = (event.xdata, event.ydata)
                self.annot.set_text(f"PRsM: {prsm_index}\nWins: {int(activation_count.item())}")
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if self.annot.get_visible():
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()
        else:
            if self.annot.get_visible():
                self.annot.set_visible(False)
                self.fig.canvas.draw_idle()

    def _on_recon_click(self, event):
        """Callback for mouse click events on the reconstructed tile."""
        if event.inaxes == self.ax_recon:
            if not self.recon_data:
                print("Please run 'Analyze Selected Tile' first.")
                return

            recon = self.recon_data.get('recon')
            firing_mask = self.recon_data.get('firing_mask')

            if recon is None or firing_mask is None:
                return

            firing_indices = torch.where(firing_mask[0] == True)[0]
            if len(firing_indices) == 0:
                print("No PRSMs fired for this tile.")
                return

            firing_recons = recon[0, firing_indices]
            
            num_recons = len(firing_recons)
            cols = int(math.ceil(math.sqrt(num_recons)))
            rows = int(math.ceil(num_recons / cols))
            
            new_fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
            new_fig.canvas.manager.set_window_title(f'Firing PRSMs ({num_recons} total)')
            
            axes = axes.flatten()

            for i, ax in enumerate(axes):
                if i < num_recons:
                    recon_img = firing_recons[i].cpu().view(3, self.tile_size, self.tile_size).permute(1, 2, 0)
                    ax.imshow(recon_img)
                    ax.set_title(f"PRSM {firing_indices[i].item()}", fontsize=8)
                ax.axis('off')

            new_fig.tight_layout()
            new_fig.show()
class PRM_L2_Visual_Debugger:
    def __init__(self, model, dataset, device, config):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.tile_size = config['tile_params']['size']
        self.stride = config['tile_params']['stride']
        self.init_recep_fld = 0
        self.firing_std_factor: float = config['firing_std_factor']
        self.firing_selection_method = config['l2']['firing_selection_method']
        self.recep_fld_size: int = config['l2']['receptive_field'][0]

        self.config = config

        self.recon_data = {}
        self.prsm_activations = {}  # Store top/bottom activations for each PRSM

        self.recep_fld_px_size = self.tile_size + (self.recep_fld_size - 1)*(self.tile_size - self.stride)
        print('recep fld px size:', self.recep_fld_px_size)
        
        self.fig = plt.figure(figsize=(18, 10))
        gs = self.fig.add_gridspec(2, 3)
        self.ax_image = self.fig.add_subplot(gs[0, 0]) #Shows the whole image that is being processed
        self.ax_recep_fld = self.fig.add_subplot(gs[1, 0]) #Shows the selected tile to process
        self.ax_recon = self.fig.add_subplot(gs[1, 1]) #Shows the reconstructed pattern from the selected tile
        self.ax_info = self.fig.add_subplot(gs[1, 2])
        self.ax_heatmap = self.fig.add_subplot(gs[0, 1:]) #Heatmap of the prsms activation rates
        
        self.fig.suptitle('PRM Layer 2 Visual Debugger', fontsize=16)

        self.ax_image.set_title("Original Image")
        self.ax_image.axis('off')
        self.ax_recep_fld.set_title("Selected Receptive Field")
        self.ax_recep_fld.axis('off')
        self.ax_recon.set_title("Reconstructed by Best PRsM")
        self.ax_recon.axis('off')
        self.ax_info.axis('off')
        self.ax_heatmap.set_title("PRsM Activation Frequency for this Image")

        plt.subplots_adjust(bottom=0.2)
        ax_img_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        ax_tile_slider = plt.axes([0.25, 0.05, 0.65, 0.03])
        ax_button_full = plt.axes([0.05, 0.1, 0.15, 0.04])
        ax_button_tile = plt.axes([0.05, 0.05, 0.15, 0.04])

        self.img_slider = Slider(ax_img_slider, 'Image Index', 0, len(self.dataset) - 1, valinit=0, valstep=1)
        self.recep_fld_slider = Slider(ax_tile_slider, 'Receptive Field Index', 0, 100, valinit=self.init_recep_fld, valstep=1)

        self.run_tile_button = Button(ax_button_tile, 'Analyze Selected Receptive Field')
        self.run_full_button = Button(ax_button_full, 'Analyze Full Image')

        self.img_slider.on_changed(self._update_display)
        self.recep_fld_slider.on_changed(self._update_display)
        
        self.run_tile_button.on_clicked(self._perform_receptive_field_forward_pass)
        self.run_full_button.on_clicked(self._perform_full_forward_pass)


        self.rect = patches.Rectangle((0, 0), self.recep_fld_px_size, self.recep_fld_px_size, linewidth=2, edgecolor='r', facecolor='none')
        self.ax_image.add_patch(self.rect)
        
        self.colorbar = None
        self.annot = self.ax_heatmap.annotate("", xy=(0,0), xytext=(20,20),
                                               textcoords="offset points",
                                               bbox=dict(boxstyle="round", fc="w"),
                                               arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)
        self.grid_shape = config['l2']['grid_shape']
        self.heatmap_data = np.zeros(self.grid_shape)

        self.cmap = plt.get_cmap('viridis').copy()
        self.cmap.set_under(color='black')
        self.extent = [-0.5, self.grid_shape[1] - 0.5, self.grid_shape[0] - 0.5, -0.5]
        self.heatmap_im = self.ax_heatmap.imshow(self.heatmap_data, cmap=self.cmap, aspect='auto', vmin=1, extent=self.extent)
        self.ax_heatmap.set_title("PRsM Activation Frequency for this Image (Black = 0 wins)")
        self.ax_heatmap.xaxis.set_major_locator(MultipleLocator(5))
        self.ax_heatmap.yaxis.set_major_locator(MultipleLocator(5))
        self.colorbar = self.fig.colorbar(self.heatmap_im, ax=self.ax_heatmap)
        
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.fig.canvas.mpl_connect('button_press_event', self._on_recon_click)
        self._update_display(0)
        plt.show()

    def _update_display(self, val):
        """Updates the display based on slider values without running the model."""
        img_idx = int(self.img_slider.val)
        
        dataset_item = self.dataset[img_idx]
        if dataset_item is None or dataset_item[0] is None:
            print(f"Warning: Could not load data for image index {img_idx}. Skipping.")
            return
            
        original_img, all_tiles, img_path = dataset_item
        
        if all_tiles is None: return
        
        num_tiles = all_tiles.shape[0]
        _, h, w = original_img.size()
        num_tiles_h = (h - self.tile_size) // self.stride + 1
        num_tiles_w = (w - self.tile_size) // self.stride + 1

        num_recep_flds_h = (num_tiles_h - self.recep_fld_size + 1)
        num_recep_flds_w = (num_tiles_w - self.recep_fld_size + 1)

        num_recep_flds = num_recep_flds_h * num_recep_flds_w

        self.recep_fld_slider.valmax = num_recep_flds - 1
        self.recep_fld_slider.ax.set_xlim(0, num_recep_flds - 1)
        
        recep_fld_idx = int(self.recep_fld_slider.val)
        if recep_fld_idx >= num_tiles:
            recep_fld_idx = num_tiles - 1
            self.recep_fld_slider.set_val(recep_fld_idx)

        self.ax_image.imshow(original_img.permute(1, 2, 0))
    
        row = (recep_fld_idx // num_recep_flds_w) * self.stride
        col = (recep_fld_idx % num_recep_flds_w) * self.stride
    
        self.rect.set_xy((col, row))

        selected_recep_fld = original_img[:, row:row+self.recep_fld_px_size, col:col+self.recep_fld_px_size]

        self.ax_recep_fld.imshow(selected_recep_fld.cpu().view(3, self.recep_fld_px_size, self.recep_fld_px_size).permute(1, 2, 0))

        self.ax_recon.clear()
        self.ax_recon.set_title("Reconstructed by Best PRsM")
        self.ax_recon.axis('off')
        self.ax_info.clear()
        self.ax_info.axis('off')
        img_filename = os.path.basename(img_path)
        info_text = (f"Image: {img_filename}\n"
                     f"Receptive field idx: {recep_fld_idx}\n\n"
                     f"Click 'Analyze' to see results.")
        self.ax_info.text(0, 0.5, info_text, fontsize=12, va='center')

        self.fig.canvas.draw_idle()

    def _apply_firing_threshold(self, errors: torch.Tensor):
        if self.firing_selection_method == FiringSelectionMethod.StdFiringThreshold:
            mean_errors = errors.mean(dim=1, keepdim=True)
            print(f"Mean errors shape: {mean_errors.shape}")
            std_errors = errors.std(dim=1, keepdim=True)
            print(f"Std errors shape: {std_errors.shape}")

            current_std_factor = self.firing_std_factor
            dynamic_threshold = mean_errors - (current_std_factor * std_errors)
            firing_mask = errors < dynamic_threshold
            # print(f"Firing mask shape: {firing_mask.shape}")

            # Iteratively lower threshold if no units fire
            for i in range(firing_mask.shape[0]):
                if firing_mask[i].sum() == 0:
                    current_std_factor = self.firing_std_factor
                    print(f"No PRSMs fired for receptive field {i}. Mean error: {mean_errors[i].item():.4f}. Std: {std_errors[i].item():.4f}. Min error: {errors[i].min().item():.4f}. Lowering threshold...")
                    dynamic_threshold = errors[i].min() + (mean_errors[i] - errors[i].min()) * 0.01
                    print(f"New dynamic threshold: {dynamic_threshold.item():.8f}")
                    firing_mask[i] = errors[i] < dynamic_threshold
                    # while firing_mask[i].sum() == 0:
                    #     current_std_factor *= 0.995
                    #     print(f"Current std factor: {current_std_factor}")
                    #     dynamic_threshold = mean_errors[i] - (current_std_factor * std_errors[i])
                    #     if (dynamic_threshold < 0):
                    #         current_std_factor = mean_errors[i] / std_errors[i]
                    #         print(f"Dynamic threshold went negative, adjusting std factor to {current_std_factor}")
                    #         continue
                    #     print(f"New dynamic threshold: {dynamic_threshold.item():.8f}")
                    #     firing_mask[i] = errors[i] < dynamic_threshold
                    #     if current_std_factor < 0.001: # Safety break
                    #         print("Warning: Could not find any firing PRSMs even after lowering threshold significantly.")
                    #         break
                print(f"Receptive field {i}: {firing_mask[i].sum().item()} PRSMs fired.")
            return firing_mask
        else: # WinnerTakeAll
            winner_indices = torch.argmin(errors, dim=1)
            firing_mask = torch.zeros_like(errors, dtype=torch.bool)
            firing_mask[torch.arange(errors.shape[0]), winner_indices] = True
            return firing_mask


    def _perform_receptive_field_forward_pass(self, event):
        """Runs a fast forward pass on only the selected receptive field."""
        img_idx = int(self.img_slider.val)
        recep_fld_idx = int(self.recep_fld_slider.val)

        dataset_item = self.dataset[img_idx]
        if dataset_item is None or dataset_item[1] is None: return

        original_img, all_tiles, img_path = dataset_item
        print(f"\nAll tiles size: {all_tiles.size()}")
        _, h, w = original_img.size()
        num_tiles_h = (h - self.tile_size) // self.stride + 1
        num_tiles_w = (w - self.tile_size) // self.stride + 1
        all_tiles = all_tiles.view(num_tiles_w, num_tiles_h, -1)
        
        num_recep_flds_h = (num_tiles_h - self.recep_fld_size + 1)
        num_recep_flds_w = (num_tiles_w - self.recep_fld_size + 1)

        row = (recep_fld_idx // num_recep_flds_w)
        col = (recep_fld_idx % num_recep_flds_w)

        reshaped_tiles = all_tiles.view(num_tiles_h, num_tiles_w, 3, self.tile_size, self.tile_size)        

        selected_tiles = reshaped_tiles[row:row+self.recep_fld_size, col:col+self.recep_fld_size]
        print(f"\nSelected tiles size: {selected_tiles.size()}")
        
        row_px = row * self.stride
        col_px = col * self.stride
        selected_img = original_img[:, row_px:row_px+self.recep_fld_px_size, col_px:col_px+self.recep_fld_px_size]
        
        activation_map = self.model.generate_l1_activation_map(selected_img.to(device), config['l2']['gen_l1_act_map_batch_size'])
        print(f"Activation map size: {activation_map.size()}")
        np.set_printoptions(linewidth=120, threshold=10000, edgeitems=13, formatter={'float_kind':lambda x: f"{x:8.4f}"})
        print("Activation map sample values (0,0):")
        print(activation_map[0, 0, :].cpu().detach().numpy(), end="\n\n")
        print("Activation map sample values (0,1):")
        print(activation_map[0, 1, :].cpu().detach().numpy(), end="\n\n")

        activation_map = activation_map.permute(2, 0, 1)

        activation_map_4d = activation_map.unsqueeze(0)

        receptive_fields = F.unfold(
            activation_map_4d,
            kernel_size=self.recep_fld_size
        )

        l2_input_receptive_fields = receptive_fields.squeeze(0).transpose(1, 0)

        activation_map_reconstructed = l2_input_receptive_fields.transpose(1, 0).view(1, self.model.layer_specs[1]['prsm_count'], 4, 4)

        print(f"\nActivation map reconstructed size: {activation_map_reconstructed.size()}")
        print(f"Activation map reconstructed sample values (0,0):")
        print(activation_map_reconstructed[:, :, 0, 0].cpu().detach().numpy(), end="\n\n")

        recon, _ = self.model.layers['layer_2'](l2_input_receptive_fields)
        print(f"L2 recon shape: {recon.shape}, L2 input receptive fields shape: {l2_input_receptive_fields.shape}")

        errors = F.mse_loss(recon, l2_input_receptive_fields.unsqueeze(0), reduction='none').mean(dim=2)
        
        winner_idx = torch.argmin(errors, dim=1).item()
        
        recon_error = errors.squeeze(0)[winner_idx]
        
        # Now let's process each tile through layer 1 and create weighted reconstructions
        print(f"\nProcessing {self.recep_fld_size}x{self.recep_fld_size} tiles through layer 1...")
        
        # Get the number of PRSMs in layer 1
        num_l1_prsms = self.model.layer_specs[1]['prsm_count']
        num_tiles = self.recep_fld_size * self.recep_fld_size
        
        print(f"Number of tiles: {num_tiles}, Number of L1 PRSMs: {num_l1_prsms}")
        
        # Reshape L2 reconstruction to (num_tiles, num_l1_prsms)
        # recon shape is (1, num_tiles * num_l1_prsms) - we need to reshape it
        winner_recon = recon[:, winner_idx, :]
        print(f"L2 winner recon shape: {winner_recon.shape}")
        l2_best_recon_reshaped = winner_recon.transpose(1, 0).view(1, self.model.layer_specs[1]['prsm_count'], self.recep_fld_size, self.recep_fld_size).squeeze(0).permute(1, 2, 0).view(num_tiles, num_l1_prsms)
        l2_best_recon_reshaped_after_activation = self._apply_firing_threshold(1-l2_best_recon_reshaped)
        
        print(f"L2 best recon reshaped size: {l2_best_recon_reshaped.size()}")
        print(f"L2 best recon after activation size: {l2_best_recon_reshaped_after_activation.size()}")
        print(f"L2 best recon sample values (before activation):")
        print(l2_best_recon_reshaped[0].cpu().detach().numpy(), end="\n\n")
        print(f"L2 best recon after activation sample values:")
        print(l2_best_recon_reshaped_after_activation[0].cpu().detach().numpy(), end="\n\n")

        recon_activated = l2_best_recon_reshaped * l2_best_recon_reshaped_after_activation.float()
        print(f"L2 best recon after activation (applied) size:")
        print(recon_activated.size())
        print(f"L2 best recon after activation (applied) sample values:")
        print(recon_activated[0].cpu().detach().numpy(), end="\n\n")

        final_tile_reconstructions = []
        
        for i in range(self.recep_fld_size):
            for j in range(self.recep_fld_size):
                tile_idx = i * self.recep_fld_size + j
                
                # Get individual tile
                tile = selected_tiles[i, j].to(self.device)  # Shape: (3, tile_size, tile_size)
                
                # Process through layer 1
                with torch.no_grad():
                    l1_recon, _ = self.model.layers['layer_1'](tile.view(self.tile_size*self.tile_size*3).unsqueeze(0))
                    l1_recon = l1_recon.squeeze(0)  # Shape: (num_l1_prsms, tile_size * tile_size * 3)
                print(f"Tile ({i},{j}) - L1 recon shape: {l1_recon.shape}")
                
                # Get L2 scores for this tile (shape: num_l1_prsms)
                l2_scores = l2_best_recon_reshaped[tile_idx]  # Shape: (num_l1_prsms,)
                
                # Normalize scores to prevent explosion, but maintain relative strengths
                l2_scores_normalized = l2_scores / (torch.sum(l2_scores) + 1e-9)
                weights = l2_scores_normalized.unsqueeze(1) # Shape: (num_l1_prsms, 1)
                
                # Calculate weighted sum of L1 reconstructions
                # weights shape (num_l1_prsms, 1) is broadcast over l1_recon shape (num_l1_prsms, tile_size * tile_size * 3)
                weighted_l1_recon = weights * l1_recon
                final_tile_recon = torch.sum(weighted_l1_recon, dim=0) # Shape: (tile_size * tile_size * 3)

                final_tile_reconstructions.append(final_tile_recon)
                
                print(f"Tile ({i},{j}) - L1 recon shape: {l1_recon.shape}, L2 scores shape: {l2_scores.shape}")
                print(f"  Final tile recon shape: {final_tile_recon.shape}")
                print(f"  L2 scores range: [{l2_scores.min().item():.4f}, {l2_scores.max().item():.4f}]")
        
        print(f"\nCompleted processing all tiles through layer 1.")
        print(f"L1 recon size: {l1_recon.shape}, Final tile reconstruction size: {len(final_tile_reconstructions)}x{final_tile_reconstructions[0].shape}")
        print(f"Final tile reconstruction sample values (tile 0):")
        print(final_tile_reconstructions[0].cpu().detach().numpy(), end="\n\n")
        # Now interpolate to create a 4x4 firing field
        # We need to consider the stride between tiles
        print(f"\nInterpolating final tile reconstructions to create firing field...")
        print(f"Tile size: {self.tile_size}, Stride: {self.stride}")
        
        # Convert final tile reconstructions to 3-channel format for interpolation
        # Each final_tile_recon is (tile_size * tile_size * 3), we need (3, tile_size, tile_size)
        final_tile_reconstructions_3ch = []
        for i, tile_recon in enumerate(final_tile_reconstructions):
            # Repeat the single channel to create 3 channels
            tile_recon_3ch = tile_recon.view(3, self.tile_size, self.tile_size)
            final_tile_reconstructions_3ch.append(tile_recon_3ch)
            print(f"Tile {i} recon 3ch shape: {tile_recon_3ch.shape}")

        print(f"Final tile reconstructions 3ch list length: {len(final_tile_reconstructions_3ch)}")
        print(f"Final tile reconstructions 3ch sample values (tile 0):")
        print(final_tile_reconstructions_3ch[0].cpu().detach().numpy(), end="\n\n")
        
        interpolated_field = self._interpolate_tile_reconstructions(final_tile_reconstructions_3ch, self.recep_fld_size, self.tile_size, self.stride) # Shape: (3, recep_fld_px_size, recep_fld_px_size)
        
        print(f"Interpolated field shape: {interpolated_field.shape}")
        
        # Display the interpolated field
        self.ax_recon.clear()
        self.ax_recon.imshow(interpolated_field.permute(1, 2, 0).cpu().detach())
        self.ax_recon.set_title(f"Receptive Field {recep_fld_idx} - Weighted Interpolated Field")
        self.ax_recon.axis('off')
        
        img_filename = os.path.basename(img_path)
        info_text = (f"Image: {img_filename}\n"
                     f"Receptive field index: {recep_fld_idx}\n\n"
                     f"Best PRsM: {winner_idx}\n"
                     f"Best Error: {recon_error:.10f}\n\n"
                     f"Interpolated field size: {interpolated_field.shape[1]}x{interpolated_field.shape[2]}")

        self.ax_info.clear()
        self.ax_info.axis('off')
        self.ax_info.text(0, 0.5, info_text, fontsize=12, va='center')

        selected_reshaped_tiles = selected_tiles.view(self.recep_fld_size, self.recep_fld_size, -1)
        print(f"\nSelected reshaped tiles size: {selected_reshaped_tiles.size()}")
        
        self.fig.canvas.draw_idle()
        return

    def _reconstruct_receptive_field_from_l2(self, l2_recon_vector, original_img, recep_fld_row, recep_fld_col):
        """
        Reconstruct a receptive field by processing L2 reconstruction through L1.
        
        Args:
            l2_recon_vector: L2 reconstruction vector for this receptive field (shape: prsm_count * rf_size * rf_size)
            original_img: Original image tensor
            recep_fld_row: Row index of the receptive field
            recep_fld_col: Column index of the receptive field
            
        Returns:
            Reconstructed receptive field as a tensor
        """
        num_l1_prsms = self.model.layer_specs[1]['prsm_count']
        num_tiles = self.recep_fld_size * self.recep_fld_size
        
        # Reshape L2 reconstruction
        l2_recon_reshaped = l2_recon_vector.view(num_l1_prsms, self.recep_fld_size, self.recep_fld_size)
        l2_recon_reshaped = l2_recon_reshaped.permute(1, 2, 0).reshape(num_tiles, num_l1_prsms)
        
        # Get the actual tiles from the original image
        row_px = recep_fld_row * self.stride
        col_px = recep_fld_col * self.stride
        selected_img = original_img[:, row_px:row_px+self.recep_fld_px_size, col_px:col_px+self.recep_fld_px_size]
        
        # Generate tiles manually
        tiles = []
        for i in range(self.recep_fld_size):
            for j in range(self.recep_fld_size):
                tile_row = i * self.stride
                tile_col = j * self.stride
                tile = selected_img[:, tile_row:tile_row+self.tile_size, tile_col:tile_col+self.tile_size]
                tiles.append(tile)
        
        # Reconstruct each tile through L1
        final_tile_reconstructions = []
        for tile_idx in range(num_tiles):
            tile = tiles[tile_idx].to(self.device)
            
            with torch.no_grad():
                l1_recon, _ = self.model.layers['layer_1'](tile.view(self.tile_size*self.tile_size*3).unsqueeze(0))
                l1_recon = l1_recon.squeeze(0)
            
            # Get L2 scores for this tile
            l2_scores = l2_recon_reshaped[tile_idx]
            
            # Normalize and weight
            l2_scores_normalized = l2_scores / (torch.sum(l2_scores) + 1e-9)
            weights = l2_scores_normalized.unsqueeze(1)
            
            # Calculate weighted sum
            weighted_l1_recon = weights * l1_recon
            final_tile_recon = torch.sum(weighted_l1_recon, dim=0)
            
            final_tile_reconstructions.append(final_tile_recon)
        
        # Convert to 3-channel format
        final_tile_reconstructions_3ch = []
        for tile_recon in final_tile_reconstructions:
            tile_recon_3ch = tile_recon.view(3, self.tile_size, self.tile_size)
            final_tile_reconstructions_3ch.append(tile_recon_3ch)
        
        # Interpolate
        interpolated_field = self._interpolate_tile_reconstructions(
            final_tile_reconstructions_3ch, 
            self.recep_fld_size, 
            self.tile_size, 
            self.stride
        )
        
        return interpolated_field

    def _perform_full_forward_pass(self, event):
        """Runs the forward pass on all tiles of the image and updates results."""
        img_idx = int(self.img_slider.val)
        
        dataset_item = self.dataset[img_idx]
        if dataset_item is None or dataset_item[1] is None: return

        image_tensor, all_tiles, _ = dataset_item

        self.model.eval()

        activation_map = self.model.generate_l1_activation_map(image_tensor.to(device), config['l2']['internal_batch_size'])
        activation_map = activation_map.permute(2, 0, 1)
        activation_map_4d = activation_map.unsqueeze(0)

        receptive_fields = F.unfold(
            activation_map_4d,
            kernel_size=self.config['l2']['receptive_field']
        )
        l2_input_receptive_fields = receptive_fields.squeeze(0).transpose(1, 0)

        all_errors_list = []
        all_recons_list = []
        with torch.no_grad():
            for receptive_field_batch in tqdm(torch.split(l2_input_receptive_fields, self.config['l2']['internal_batch_size'])):
                with autocast():
                    recon, _ = self.model.layers['layer_2'](receptive_field_batch)
                    errors = F.mse_loss(recon, receptive_field_batch.unsqueeze(1), reduction='none').mean(dim=2)             
                    all_errors_list.append(errors)
                    all_recons_list.append(recon)
            all_errors = torch.cat(all_errors_list, dim=0)
            all_recons = torch.cat(all_recons_list, dim=0)
            
            firing_selection_method = self.firing_selection_method
            self.firing_selection_method = FiringSelectionMethod.WinnerTakeAll
            firing_mask = self._apply_firing_threshold(all_errors)
            
            # Calculate receptive field positions
            _, h, w = image_tensor.size()
            num_tiles_h = (h - self.tile_size) // self.stride + 1
            num_tiles_w = (w - self.tile_size) // self.stride + 1
            num_recep_flds_h = (num_tiles_h - self.recep_fld_size + 1)
            num_recep_flds_w = (num_tiles_w - self.recep_fld_size + 1)
            
            # Store top 10 and bottom 10 activations for each PRSM
            num_prsms = all_errors.shape[1]
            self.prsm_activations = {}
            
            print("Computing top and bottom activations for each PRSM...")
            for prsm_idx in tqdm(range(num_prsms)):
                prsm_errors = all_errors[:, prsm_idx]
                prsm_firing_mask = firing_mask[:, prsm_idx]
                
                # Get indices where this PRSM fired
                fired_indices = torch.where(prsm_firing_mask)[0]
                
                if len(fired_indices) == 0:
                    continue
                
                # Get errors for fired activations
                fired_errors = prsm_errors[fired_indices]
                
                # Top 10 (lowest errors)
                num_top = min(10, len(fired_errors))
                top_k_values, top_k_local_indices = torch.topk(fired_errors, num_top, largest=False)
                top_k_indices = fired_indices[top_k_local_indices]
                
                # Bottom 10 (highest errors among fired)
                num_bottom = min(10, len(fired_errors))
                bottom_k_values, bottom_k_local_indices = torch.topk(fired_errors, num_bottom, largest=True)
                bottom_k_indices = fired_indices[bottom_k_local_indices]
                
                # Store activation data
                top_activations = []
                for idx, error in zip(top_k_indices, top_k_values):
                    rf_idx = idx.item()
                    row = rf_idx // num_recep_flds_w
                    col = rf_idx % num_recep_flds_w
                    
                    # Get original receptive field
                    row_px = row * self.stride
                    col_px = col * self.stride
                    original_rf = image_tensor[:, row_px:row_px+self.recep_fld_px_size, col_px:col_px+self.recep_fld_px_size]
                    
                    # Reconstruct receptive field from L2
                    l2_recon_vector = all_recons[rf_idx, prsm_idx, :]
                    reconstructed_rf = self._reconstruct_receptive_field_from_l2(
                        l2_recon_vector, image_tensor, row, col
                    )
                    
                    top_activations.append({
                        'rf_idx': rf_idx,
                        'error': error.item(),
                        'original': original_rf.cpu(),
                        'reconstructed': reconstructed_rf.cpu()
                    })
                
                bottom_activations = []
                for idx, error in zip(bottom_k_indices, bottom_k_values):
                    rf_idx = idx.item()
                    row = rf_idx // num_recep_flds_w
                    col = rf_idx % num_recep_flds_w
                    
                    # Get original receptive field
                    row_px = row * self.stride
                    col_px = col * self.stride
                    original_rf = image_tensor[:, row_px:row_px+self.recep_fld_px_size, col_px:col_px+self.recep_fld_px_size]
                    
                    # Reconstruct receptive field from L2
                    l2_recon_vector = all_recons[rf_idx, prsm_idx, :]
                    reconstructed_rf = self._reconstruct_receptive_field_from_l2(
                        l2_recon_vector, image_tensor, row, col
                    )
                    
                    bottom_activations.append({
                        'rf_idx': rf_idx,
                        'error': error.item(),
                        'original': original_rf.cpu(),
                        'reconstructed': reconstructed_rf.cpu()
                    })
                
                self.prsm_activations[prsm_idx] = {
                    'top': top_activations,
                    'bottom': bottom_activations
                }
            


            self.firing_selection_method = firing_selection_method
            win_counts = firing_mask.sum(dim=0).float()

            self.heatmap_data = win_counts.reshape(self.grid_shape).cpu()
            
            self.heatmap_im.set_data(self.heatmap_data)
            
            max_val = self.heatmap_data.max()
            self.heatmap_im.set_clim(vmin=1, vmax=max_val if max_val > 0 else 1)
            
            if max_val > 0:
                tick_step = max(1, int(np.ceil(max_val / 10)))
                self.colorbar.set_ticks(np.arange(0, max_val + 1, tick_step))
            else:
                self.colorbar.set_ticks([])

            self.fig.canvas.draw_idle()

    def _on_hover(self, event):
        """Callback for mouse hover events on the heatmap."""
        if event.inaxes == self.ax_heatmap:
            col = int(round(event.xdata))
            row = int(round(event.ydata))

            if 0 <= row < self.grid_shape[0] and 0 <= col < self.grid_shape[1]:
                prsm_index = row * self.grid_shape[1] + col
                activation_count = self.heatmap_data[row, col]
                
                self.annot.xy = (event.xdata, event.ydata)
                self.annot.set_text(f"PRsM: {prsm_index}\nWins: {int(activation_count.item())}")
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if self.annot.get_visible():
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()
        else:
            if self.annot.get_visible():
                self.annot.set_visible(False)
                self.fig.canvas.draw_idle()

    def _on_recon_click(self, event):
        """Callback for mouse click events on the heatmap to show top/bottom activations."""
        if event.inaxes == self.ax_heatmap:
            # Check if we have stored activation data
            if not self.prsm_activations:
                print("Please run 'Analyze Full Image' first to see PRSM activations.")
                return
            
            # Get the clicked PRSM index
            col = int(round(event.xdata))
            row = int(round(event.ydata))
            
            if 0 <= row < self.grid_shape[0] and 0 <= col < self.grid_shape[1]:
                prsm_index = row * self.grid_shape[1] + col
                
                if prsm_index not in self.prsm_activations:
                    print(f"PRSM {prsm_index} has no activations.")
                    return
                
                prsm_data = self.prsm_activations[prsm_index]
                top_acts = prsm_data['top']
                bottom_acts = prsm_data['bottom']
                
                # Create a new figure to display activations
                num_total = len(top_acts) + len(bottom_acts)
                cols_per_row = 5
                rows = int(np.ceil(num_total / cols_per_row))
                
                new_fig, axes = plt.subplots(rows, cols_per_row * 2, figsize=(cols_per_row * 4, rows * 2))
                new_fig.canvas.manager.set_window_title(f'PRSM {prsm_index} - Top {len(top_acts)} and Bottom {len(bottom_acts)} Activations')
                new_fig.suptitle(f'PRSM {prsm_index} Activations', fontsize=14)
                
                if rows == 1:
                    axes = axes.reshape(1, -1)
                axes = axes.flatten()
                
                # Display top activations
                for i, act in enumerate(top_acts):
                    # Original
                    ax_orig = axes[i * 2]
                    orig_img = act['original'].permute(1, 2, 0)
                    ax_orig.imshow(orig_img)
                    ax_orig.set_title(f"Top {i+1}\nRF {act['rf_idx']}\nErr: {act['error']:.6f}", fontsize=8)
                    ax_orig.axis('off')
                    
                    # Reconstructed
                    ax_recon = axes[i * 2 + 1]
                    recon_img = act['reconstructed'].permute(1, 2, 0)
                    ax_recon.imshow(recon_img)
                    ax_recon.set_title("Reconstructed", fontsize=8)
                    ax_recon.axis('off')
                
                # Display bottom activations
                offset = len(top_acts) * 2
                for i, act in enumerate(bottom_acts):
                    # Original
                    ax_orig = axes[offset + i * 2]
                    orig_img = act['original'].permute(1, 2, 0)
                    ax_orig.imshow(orig_img)
                    ax_orig.set_title(f"Bottom {i+1}\nRF {act['rf_idx']}\nErr: {act['error']:.6f}", fontsize=8)
                    ax_orig.axis('off')
                    
                    # Reconstructed
                    ax_recon = axes[offset + i * 2 + 1]
                    recon_img = act['reconstructed'].permute(1, 2, 0)
                    ax_recon.imshow(recon_img)
                    ax_recon.set_title("Reconstructed", fontsize=8)
                    ax_recon.axis('off')
                
                # Hide unused axes
                for i in range(num_total * 2, len(axes)):
                    axes[i].axis('off')
                
                new_fig.tight_layout()
                new_fig.show()
        
        elif event.inaxes == self.ax_recon:
            # Keep the old functionality for clicking on reconstruction
            if not self.recon_data:
                print("Please run 'Analyze Selected Tile' first.")
                return

            recon = self.recon_data.get('recon')
            firing_mask = self.recon_data.get('firing_mask')

            if recon is None or firing_mask is None:
                return

            firing_indices = torch.where(firing_mask[0] == True)[0]
            if len(firing_indices) == 0:
                print("No PRSMs fired for this tile.")
                return

            firing_recons = recon[0, firing_indices]
            
            num_recons = len(firing_recons)
            cols = int(math.ceil(math.sqrt(num_recons)))
            rows = int(math.ceil(num_recons / cols))
            
            new_fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
            new_fig.canvas.manager.set_window_title(f'Firing PRSMs ({num_recons} total)')
            
            axes = axes.flatten()

            for i, ax in enumerate(axes):
                if i < num_recons:
                    recon_img = firing_recons[i].cpu().view(3, self.tile_size, self.tile_size).permute(1, 2, 0)
                    ax.imshow(recon_img)
                    ax.set_title(f"PRSM {firing_indices[i].item()}", fontsize=8)
                ax.axis('off')

            new_fig.tight_layout()
            new_fig.show()

    def _interpolate_tile_reconstructions(self, weighted_reconstructions: list[torch.Tensor], recep_fld_size: int, tile_size: int, stride: int):
        """
        Interpolate weighted tile reconstructions to create a continuous firing field.
        
        Args:
            weighted_reconstructions: List of weighted reconstructions for each tile
            recep_fld_size: Size of SQUARE receptive field in tiles (e.g., 4x4)
            tile_size: Size of each tile
            stride: Stride between tiles
            
        Returns:
            Interpolated field tensor
        """
        import torch.nn.functional as F
        
        # Calculate the output size considering stride
        # If stride < tile_size, there will be overlap
        output_size = (recep_fld_size - 1) * stride + tile_size
        print(f"Interpolating to output size: {output_size}x{output_size}")
        
        # Create a canvas for the interpolated field
        canvas = torch.zeros(3, output_size, output_size, device=weighted_reconstructions[0].device)
        weight_canvas = torch.zeros(output_size, output_size, device=weighted_reconstructions[0].device)
        
        # Place each weighted reconstruction on the canvas
        for i in range(recep_fld_size):
            for j in range(recep_fld_size):
                tile_idx = i * recep_fld_size + j
                recon = weighted_reconstructions[tile_idx]
                
                # Calculate position on canvas
                start_h = i * stride
                start_w = j * stride
                end_h = start_h + tile_size
                end_w = start_w + tile_size
                
                # Add the reconstruction to the canvas
                canvas[:, start_h:end_h, start_w:end_w] += recon
                
                # Track weights for normalization
                weight_canvas[start_h:end_h, start_w:end_w] += 1.0
        
        # Normalize by the number of overlapping tiles
        # Avoid division by zero
        weight_canvas = torch.clamp(weight_canvas, min=1.0)
        
        # Normalize the canvas
        for c in range(3):
            canvas[c] = canvas[c] / weight_canvas
        
        return canvas

class PRM_Debugger_App:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.prm_model = None
        self.visualization_dataset = None

        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.fig.canvas.manager.set_window_title('PRM Debugger')
        self.ax.set_title('Select a Layer to Debug')
        self.ax.axis('off')

        ax_l1_button = plt.axes([0.1, 0.6, 0.8, 0.2])
        ax_l2_button = plt.axes([0.1, 0.3, 0.8, 0.2])
        
        self.l1_button = Button(ax_l1_button, 'Debug Layer 1')
        self.l2_button = Button(ax_l2_button, 'Debug Layer 2')
        
        self.l1_button.on_clicked(self.launch_l1_debugger)
        self.l2_button.on_clicked(self.launch_l2_debugger)
        
        plt.show()

    def _load_model_and_data(self):
        if self.prm_model is None:
            print("Loading model and data for the first time...")
            data_csv_path, data_root_path, save_dir = env_config(self.config)
            
            self.prm_model = PRM(
                tile_params=self.config['tile_params'],
                l2_receptive_field=self.config['l2']['receptive_field'],
                layers_to_load=[1, 2], # Load both for potential L2 debug
                firing_std_factor=self.config['firing_std_factor'],
                conscience_factor=self.config['conscience_factor'],
                win_rate_momentum=self.config['win_rate_momentum']
            )
            
            l1_checkpoint_path = os.path.join(save_dir, "prm_layer1_latest.pth")
            if os.path.exists(l1_checkpoint_path):
                self.prm_model.load_layer_weights(1, l1_checkpoint_path, self.device)

            l2_checkpoint_path = os.path.join(save_dir, "prm_layer2_latest.pth")
            if os.path.exists(l2_checkpoint_path):
                self.prm_model.load_layer_weights(2, l2_checkpoint_path, self.device)

            self.prm_model.to(self.device)
            
            transform = T.Compose([T.ToTensor()])
            self.visualization_dataset = ImageTileDataset(
                csv_file=data_csv_path, 
                data_root=data_root_path, 
                transform=transform, 
                frame_skip=9, 
                tile_params=self.config['tile_params']
            )
        
    def launch_l1_debugger(self, event):
        self._load_model_and_data()
        print("\n--- Launching Layer 1 Visual Debugger ---")
        debugger = PRM_L1_Visual_Debugger(
            self.prm_model, 
            self.visualization_dataset, 
            self.device, 
            self.config,
        )

    def launch_l2_debugger(self, event):
        self._load_model_and_data()
        print("\n--- Launching Layer 2 Visual Debugger (Dummy) ---")
        debugger = PRM_L2_Visual_Debugger(
            self.prm_model, 
            self.visualization_dataset, 
            self.device, 
            self.config
        )



def env_config(config):
    if config["run_in_colab"]:
        gdrive_root = '/content/drive/MyDrive'
        project_root = os.path.join(gdrive_root, config["project_folder"])

        drive_zip_path = os.path.join(project_root, 'data', config['zip_file_name'])
        local_data_dir = config['local_data_dir']

        if not os.path.exists(local_data_dir):
            print(f"Local data not found. Unzipping from {drive_zip_path}...")
            os.makedirs(local_data_dir, exist_ok=True)
            os.system(f'unzip -q "{drive_zip_path}" -d "{local_data_dir}"')
            print("Unzipping complete.")
        else:
            print("Local data directory already exists.")

        data_root_path = os.path.join(local_data_dir, 'data')
    else:
        project_root = "./"
        data_root_path = os.path.join(project_root, 'data')

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    data_csv_path = os.path.join(data_root_path, 'data.csv')
    save_dir = os.path.join(project_root, 'models/PRM/saved_models')
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(data_csv_path):
        raise ValueError(f"Error: data.csv not found at {data_csv_path}.")

    return data_csv_path, data_root_path, save_dir

if __name__ == "__main__":

    config = {
        "l1": {
            "training": {
                'train': False,
                'save_interval': 1000,
                'debug_interval': 1,
                'debug_heatmap': True,
                'debug_heatmap_interval': 100,
                'save': True
            },
            'prune': False,
            'prune_threshold': 1e-6,
            'frame_skip': 2,
            "batch_size": 4400,
            "num_epochs": 1,
            "firing_selection_method": FiringSelectionMethod.WinnerTakeAll,
            "start_at_img": 0,
            "grid_shape": (13, 31)
        },
        "l2": {
            "training": {
                'train': True,
                'save_interval': 1000,
                'debug_interval': 1,
                'debug_heatmap': True,
                'debug_heatmap_interval': 16,
                'save': True,
            },
            'prune': False,
            'prune_threshold': 1e-6,
            'frame_skip': 4,
            "batch_size": 1,
            "internal_batch_size": 330,
            "gen_l1_act_map_batch_size": 377,
            "num_epochs": 1,
            "receptive_field": (4, 4),
            "firing_selection_method": FiringSelectionMethod.StdFiringThreshold,
            "start_at_img": 0,
            "grid_shape": (1, 73)
        },
        "layers_to_load": [1, 2],
        "tile_params": {'size': 16, 'stride': 8},
        "firing_std_factor": 1,
        "conscience_factor": 15,
        "win_rate_momentum": 0.995,
        "run_in_colab": 'google.colab' in sys.modules,
        "project_folder": "Neuro and AI/Thesis documents/model",
        "zip_file_name": "data.zip",
        "local_data_dir": "/content/data",
        "run_debugger": True,
        "image_size": (320, 240)
    }

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device (NVIDIA GPU) for acceleration.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon GPU) for acceleration.")
    else:
        device = torch.device("cpu")
        print("No GPU available. Using CPU.")

    _, _, _ = env_config(config)

    from models.PRM.prm import PRM
    from utils.dataset.image_tile_dataset import ImageTileDataset
    from utils import memory_monitoring


    print("\n--- Launching Visual Debugger ---")
    debugger = PRM_Debugger_App(config, device)
