import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import sys
from enum import Enum
from torch.cuda.amp import autocast, GradScaler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.image_utils import ImageUtils

class PRsM_L(nn.Module):
    """Vectorized implementation of a Layer, containing all PRsMs."""
    def __init__(self, prsm_count, input_dim, latent_dim, hidden_dims):
        super().__init__()
        self.prsm_count = prsm_count
        # Encoder
        self.enc_fc1_w = nn.Parameter(torch.randn(prsm_count, hidden_dims[0], input_dim))
        self.enc_fc1_b = nn.Parameter(torch.randn(prsm_count, hidden_dims[0]))
        self.enc_fc2_w = nn.Parameter(torch.randn(prsm_count, hidden_dims[1], hidden_dims[0]))
        self.enc_fc2_b = nn.Parameter(torch.randn(prsm_count, hidden_dims[1]))
        self.enc_fc3_w = nn.Parameter(torch.randn(prsm_count, hidden_dims[2], hidden_dims[1]))
        self.enc_fc3_b = nn.Parameter(torch.randn(prsm_count, hidden_dims[2]))
        self.enc_fc4_w = nn.Parameter(torch.randn(prsm_count, hidden_dims[3], hidden_dims[2]))
        self.enc_fc4_b = nn.Parameter(torch.randn(prsm_count, hidden_dims[3]))
        self.enc_fc5_w = nn.Parameter(torch.randn(prsm_count, latent_dim, hidden_dims[3]))
        self.enc_fc5_b = nn.Parameter(torch.randn(prsm_count, latent_dim))
        # Decoder
        self.dec_fc1_w = nn.Parameter(torch.randn(prsm_count, hidden_dims[3], latent_dim))
        self.dec_fc1_b = nn.Parameter(torch.randn(prsm_count, hidden_dims[3]))
        self.dec_fc2_w = nn.Parameter(torch.randn(prsm_count, hidden_dims[2], hidden_dims[3]))
        self.dec_fc2_b = nn.Parameter(torch.randn(prsm_count, hidden_dims[2]))
        self.dec_fc3_w = nn.Parameter(torch.randn(prsm_count, hidden_dims[1], hidden_dims[2]))
        self.dec_fc3_b = nn.Parameter(torch.randn(prsm_count, hidden_dims[1]))
        self.dec_fc4_w = nn.Parameter(torch.randn(prsm_count, hidden_dims[0], hidden_dims[1]))
        self.dec_fc4_b = nn.Parameter(torch.randn(prsm_count, hidden_dims[0]))
        self.dec_fc5_w = nn.Parameter(torch.randn(prsm_count, input_dim, hidden_dims[0]))
        self.dec_fc5_b = nn.Parameter(torch.randn(prsm_count, input_dim))
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'w' in name: nn.init.kaiming_uniform_(param, a=np.sqrt(5))
            elif 'b' in name:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(param.unsqueeze(-1))
                bound = 1 / np.sqrt(fan_in); nn.init.uniform_(param, -bound, bound)

    def forward(self, x):
        x_exp = x.unsqueeze(1)
        # Encoder
        enc1 = F.relu(torch.einsum('bpi,pih->bph', x_exp, self.enc_fc1_w.transpose(1, 2)) + self.enc_fc1_b)
        enc2 = F.relu(torch.einsum('bph,phl->bpl', enc1, self.enc_fc2_w.transpose(1, 2)) + self.enc_fc2_b)
        enc3 = F.relu(torch.einsum('bpl,plh->bph', enc2, self.enc_fc3_w.transpose(1, 2)) + self.enc_fc3_b)
        enc4 = F.relu(torch.einsum('bph,phi->bpi', enc3, self.enc_fc4_w.transpose(1, 2)) + self.enc_fc4_b)
        latent = torch.einsum('bpi,pil->bpl', enc4, self.enc_fc5_w.transpose(1, 2)) + self.enc_fc5_b
        # Decoder
        dec1 = F.relu(torch.einsum('bpl,pli->bpi', latent, self.dec_fc1_w.transpose(1, 2)) + self.dec_fc1_b)
        dec2 = F.relu(torch.einsum('bpi,pih->bph', dec1, self.dec_fc2_w.transpose(1, 2)) + self.dec_fc2_b)
        dec3 = F.relu(torch.einsum('bph,phl->bpl', dec2, self.dec_fc3_w.transpose(1, 2)) + self.dec_fc3_b)
        dec4 = F.relu(torch.einsum('bpl,plh->bph', dec3, self.dec_fc4_w.transpose(1, 2)) + self.dec_fc4_b)
        recon = torch.sigmoid(torch.einsum('bph,phi->bpi', dec4, self.dec_fc5_w.transpose(1, 2)) + self.dec_fc5_b)
        return recon, latent

class FiringSelectionMethod(Enum):
    WinnerTakeAll = 1
    StdFiringThreshold = 2

class PRM(nn.Module):
    """The full 4-layer Pattern Recognition Module. Firing selection method added"""
    def __init__(self, tile_params, l2_receptive_field=(3, 3), layers_to_load=None, win_rate_momentum=0.99, firing_std_factor=1.0, conscience_factor=10.0, firing_selection_method = FiringSelectionMethod.WinnerTakeAll):
        super().__init__()
        self.firing_std_factor = firing_std_factor
        self.tile_params = tile_params
        self.l2_receptive_field = l2_receptive_field
        self.image_utils = ImageUtils()
        self.firing_selection_method = firing_selection_method

        self.layer_specs = {
            1: {'prsm_count': 403, 'latent_dim': 2, 'hidden_dims': [24, 18, 12, 8],    'sources': [0],    'type': 'L1'},
            2: {'prsm_count': 73,  'latent_dim': 2, 'hidden_dims': [256, 64, 24, 12],   'sources': [1],    'type': 'L2'},
        }

        if layers_to_load is None: layers_to_load = self.layer_specs.keys()
        print(f"Initializing PRM with layers: {list(layers_to_load)}")
        self.layers = nn.ModuleDict()
        self.optimizers = {}

        max_layer = 0
        if layers_to_load: max_layer = max(layers_to_load)

        self.win_rate_momentum = win_rate_momentum
        self.conscience_factor = conscience_factor

        for i in range(1, max_layer + 1):
            if i not in self.layer_specs: continue
            spec = self.layer_specs[i]
            if i in layers_to_load:
                layer_key = f'layer_{i}'
                if spec['type'] == 'L1':
                    l1_input_dim = tile_params['size']**2 * 3
                    self.layers[layer_key] = PRsM_L(spec['prsm_count'], l1_input_dim, spec['latent_dim'], spec['hidden_dims'])
                    self.register_buffer('l1_win_rates', torch.ones(spec['prsm_count']) / spec['prsm_count'])
                    print(f"L1: {spec['prsm_count']} vectorized PRsMs created.")
                elif spec['type'] == 'L2':
                    l1_spec = self.layer_specs[1]
                    l2_input_dim = l2_receptive_field[0] * l2_receptive_field[1] * l1_spec['prsm_count']
                    self.layers[layer_key] = PRsM_L(spec['prsm_count'], l2_input_dim, spec['latent_dim'], spec['hidden_dims'])
                    self.register_buffer('l2_win_rates', torch.ones(spec['prsm_count']) / spec['prsm_count'])
                    print(f"L2: {spec['prsm_count']} vectorized PRsMs created.")
                else:
                    print(f"{spec['type']} not implemented yet.")
                self.optimizers[layer_key] = optim.Adam(self.layers[layer_key].parameters(), lr=1e-4)

    def generate_l1_activation_map(self, image_tensor, internal_batch_size=128):
        """
        Generates the L1 Activation Spectrum Map for a single full image.
        Processes tiles in mini-batches to conserve memory.
        """
        self.layers['layer_1'].eval()
        with torch.no_grad():
            tiles, h_tiles, w_tiles = self.image_utils.get_image_tiles(
                image_tensor, self.tile_params['size'], self.tile_params['stride'], return_indices=True
            )
            tiles = tiles.to(image_tensor.device)
            tiles = tiles / 255.0 if tiles.max() > 1.0 else tiles

            all_scores = []
            for tile_batch in torch.split(tiles, internal_batch_size):
                all_reconstructed, _ = self.layers['layer_1'](tile_batch)
                target = tile_batch.unsqueeze(1).expand(-1, all_reconstructed.size(1), -1)
                errors = F.mse_loss(all_reconstructed, target, reduction='none').mean(dim=2)

                mean_errors = errors.mean(dim=1, keepdim=True)
                std_errors = errors.std(dim=1, keepdim=True)
                dynamic_threshold = mean_errors - (self.firing_std_factor * std_errors)
                firing_mask = errors < dynamic_threshold

                scores = 1.0 - (errors)
                scores[~firing_mask] = 0
                all_scores.append(scores)

            final_scores = torch.cat(all_scores, dim=0)
            activation_map = final_scores.view(h_tiles, w_tiles, -1)
            return activation_map

    def train_layer1_batch(self, tiles_batch, scaler):
        self.layers['layer_1'].train()
        self.optimizers['layer_1'].zero_grad()

        with autocast():
            all_reconstructed, _ = self.layers['layer_1'](tiles_batch)
            errors = F.mse_loss(all_reconstructed, tiles_batch.unsqueeze(1), reduction='none').mean(dim=2)

        with torch.no_grad():
            bias = (self.l1_win_rates - (1.0 / self.layer_specs[1]['prsm_count'])) * self.conscience_factor
            biased_errors = errors + bias
            if self.firing_selection_method == FiringSelectionMethod.StdFiringThreshold:
                mean_errors = biased_errors.mean(dim=1, keepdim=True)
                std_errors = biased_errors.std(dim=1, keepdim=True)

                current_std_factor = self.firing_std_factor
                dynamic_threshold = mean_errors - (current_std_factor * std_errors)
                firing_mask = biased_errors < dynamic_threshold

                while firing_mask.sum() == 0:
                    current_std_factor -= 0.1
                    dynamic_threshold = mean_errors - (current_std_factor * std_errors)
                    firing_mask = biased_errors < dynamic_threshold
                    if current_std_factor < -5: break

                batch_win_counts = firing_mask.sum(dim=0).float()
                self.l1_win_rates.mul_(self.win_rate_momentum).add_(batch_win_counts / tiles_batch.shape[0], alpha=1.0 - self.win_rate_momentum)

                firing_errors = errors[firing_mask]
                if firing_errors.numel() == 0:
                    return 0.0
            else:
                winner_indices = torch.argmin(biased_errors, dim=1)
                batch_win_counts = torch.bincount(winner_indices, minlength=self.layer_specs[1]['prsm_count']).float()
                batch_win_rates = batch_win_counts / tiles_batch.shape[0]

                self.l1_win_rates = self.win_rate_momentum * self.l1_win_rates + (1.0 - self.win_rate_momentum) * batch_win_rates

        if self.firing_selection_method == FiringSelectionMethod.StdFiringThreshold:
            loss = firing_errors.mean()
        else:
            winner_errors = errors[torch.arange(errors.shape[0]), winner_indices]
            loss = winner_errors.mean()

        scaler.scale(loss).backward()
        scaler.step(self.optimizers['layer_1'])
        scaler.update()

        return loss.item()

    def train_layer2_batch(self, image, scaler, gen_l1_act_map_batch_size = 128, internal_batch_size=64):
        self.layers['layer_1'].eval()
        self.layers['layer_2'].train()
        self.optimizers['layer_2'].zero_grad()

        activation_map = self.generate_l1_activation_map(image, internal_batch_size=gen_l1_act_map_batch_size)
        map_h, map_w, l1_dim = activation_map.shape
        activation_map = activation_map.permute(2, 0, 1)
        activation_map_4d = activation_map.unsqueeze(0)

        receptive_fields = F.unfold(
            activation_map_4d,
            kernel_size=self.l2_receptive_field
        )
        l2_input_receptive_fields = receptive_fields.squeeze(0).transpose(1, 0)

        total_loss = 0
        total_win_counts = torch.zeros(self.layer_specs[2]['prsm_count'], device=image.device)
        num_chunks = 0

        for receptive_field_batch in torch.split(l2_input_receptive_fields, internal_batch_size):
            num_chunks += 1
            with autocast():
                recon, _ = self.layers['layer_2'](receptive_field_batch)
                errors = F.mse_loss(recon, receptive_field_batch.unsqueeze(1), reduction='none').mean(dim=2)

            with torch.no_grad():
                bias = (self.l2_win_rates - (1.0 / self.layer_specs[2]['prsm_count'])) * self.conscience_factor
                biased_errors = errors + bias

                if self.firing_selection_method == FiringSelectionMethod.StdFiringThreshold:
                    mean_errors = biased_errors.mean(dim=1, keepdim=True)
                    std_errors = biased_errors.std(dim=1, keepdim=True)
                    dynamic_threshold = mean_errors - (self.firing_std_factor * std_errors)
                    firing_mask = biased_errors < dynamic_threshold

                    while firing_mask.sum() == 0:
                        self.firing_std_factor -= 0.1
                        dynamic_threshold = mean_errors - (self.firing_std_factor * std_errors)
                        firing_mask = biased_errors < dynamic_threshold
                        if self.firing_std_factor < -5: break

                    total_win_counts += firing_mask.sum(dim=0).float()

                else:
                    winner_indices = torch.argmin(biased_errors, dim=1)
                    win_counts = torch.bincount(winner_indices, minlength=self.layer_specs[2]['prsm_count']).float()
                    total_win_counts += win_counts
                    firing_mask = torch.zeros_like(biased_errors, dtype=torch.bool)
                    firing_mask[torch.arange(receptive_field_batch.shape[0]), winner_indices] = True

            firing_errors = errors[firing_mask]
            if firing_errors.numel() == 0: continue

            loss_chunk = firing_errors.mean()
            total_loss += loss_chunk.item()
            scaler.scale(loss_chunk).backward()


        scaler.step(self.optimizers['layer_2'])
        scaler.update()

        with torch.no_grad():
            if self.firing_selection_method == FiringSelectionMethod.StdFiringThreshold:
                self.l2_win_rates.mul_(self.win_rate_momentum).add_(total_win_counts / l2_input_receptive_fields.shape[0], alpha=1.0 - self.win_rate_momentum)
            else:
                win_rates = total_win_counts / l2_input_receptive_fields.shape[0]
                self.l2_win_rates = self.win_rate_momentum * self.l2_win_rates + (1.0 - self.win_rate_momentum) * win_rates
        return total_loss / num_chunks if num_chunks > 0 else 0

    def save_layer_weights(self, layer_num, path):
        layer_key = f'layer_{layer_num}'
        if layer_key not in self.layers:
            print(f"Warning: Layer {layer_num} not found in model. Skipping save.")
            return

        state_to_save = {
            'model_state_dict': self.layers[layer_key].state_dict(),
            'optimizer_state_dict': self.optimizers[layer_key].state_dict()
        }

        # Explicitly save the win_rates buffer for the corresponding layer
        if layer_num == 1 and hasattr(self, 'l1_win_rates'):
            state_to_save['win_rates'] = self.l1_win_rates
        elif layer_num == 2 and hasattr(self, 'l2_win_rates'):
            state_to_save['win_rates'] = self.l2_win_rates

        torch.save(state_to_save, path)
        print(f"\nSaved Layer {layer_num} weights to {path}")

    def load_layer_weights(self, layer_num, path, device):
        layer_key = f'layer_{layer_num}'
        if layer_key not in self.layers:
            print(f"Warning: Layer {layer_num} not found in model. Skipping load.")
            return
        if not os.path.exists(path):
            print(f"Warning: Checkpoint for Layer {layer_num} not found at {path}. Starting from scratch.")
            return

        checkpoint = torch.load(path, map_location=device)
        self.layers[layer_key].load_state_dict(checkpoint['model_state_dict'])
        self.optimizers[layer_key].load_state_dict(checkpoint['optimizer_state_dict'])

        # Explicitly load the win_rates buffer if it exists in the checkpoint
        if 'win_rates' in checkpoint:
            if layer_num == 1:
                self.l1_win_rates.copy_(checkpoint['win_rates'])
            elif layer_num == 2:
                self.l2_win_rates.copy_(checkpoint['win_rates'])

        # Ensure optimizer state is on the correct device
        for state in self.optimizers[layer_key].state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f"Loaded Layer {layer_num} weights and optimizer states from {path}")

    def prune_layer(self, layer_num, pruning_threshold=1e-6):
        """
        Prunes inactive PRSMs from a specified layer based on their win rates.
        """
        layer_key = f'layer_{layer_num}'
        if layer_key not in self.layers:
            print(f"Error: Layer {layer_num} not found.")
            return

        win_rates = self.l1_win_rates if layer_num == 1 else self.l2_win_rates

        active_indices = torch.where(win_rates > pruning_threshold)[0]
        num_active = len(active_indices)

        if num_active == self.layer_specs[layer_num]['prsm_count']:
            print(f"Layer {layer_num}: No PRSMs to prune. All are active.")
            return

        print(f"Pruning Layer {layer_num} from {self.layer_specs[layer_num]['prsm_count']} to {num_active} PRSMs...")

        old_layer = self.layers[layer_key]
        spec = self.layer_specs[layer_num]

        # Create a new, smaller layer
        new_layer = PRsM_L(num_active, old_layer.enc_fc1_w.shape[2], spec['latent_dim'], spec['hidden_dims']).to(next(old_layer.parameters()).device)

        # Copy the weights of the active PRSMs
        with torch.no_grad():
            for old_param_name, old_param in old_layer.named_parameters():
                new_param = getattr(new_layer, old_param_name)
                new_param.data.copy_(old_param.data[active_indices])

        # Replace the old layer and update specs
        self.layers[layer_key] = new_layer
        self.layer_specs[layer_num]['prsm_count'] = num_active

        # Update the corresponding win_rates buffer
        if layer_num == 1:
            self.l1_win_rates = nn.Parameter(self.l1_win_rates.data[active_indices], requires_grad=False)
        elif layer_num == 2:
            self.l2_win_rates = nn.Parameter(self.l2_win_rates.data[active_indices], requires_grad=False)

        # Re-initialize the optimizer for the pruned layer
        self.optimizers[layer_key] = optim.Adam(self.layers[layer_key].parameters(), lr=1e-4)

        print(f"Layer {layer_num} pruned successfully.")
        MemoryMonitoring.print_model_size(self)
