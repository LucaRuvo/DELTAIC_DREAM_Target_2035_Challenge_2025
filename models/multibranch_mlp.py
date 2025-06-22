import torch
import torch.nn as nn
from utils.utils import FP_DIM

class MultiBranchMLP(nn.Module):
    def __init__(self, 
                 fp_input_dims, 
                 branch_hidden_dims, 
                 branch_embedding_dim, 
                 common_hidden_dims, 
                 dropout_rate, 
                 fp_list):
        super().__init__()
        self.fp_list = fp_list
        self.common_hidden_dims = common_hidden_dims  # Store for later access
        self.branches = nn.ModuleDict()

        # GroupNorm is used instead of BatchNorm for better performance on small batch sizes
        # Initialize each branch with its own input dimension
        for fp_name in self.fp_list:
            current_input_dim = fp_input_dims.get(fp_name, FP_DIM)  # Default to global FP_DIM
            branch_layers = []
            # First hidden layer of the branch
            branch_layers.append(nn.Linear(current_input_dim, branch_hidden_dims[0]))
            num_groups_1 = min(32, branch_hidden_dims[0] // 4)
            if num_groups_1 == 0:
                num_groups_1 = 1
            branch_layers.append(nn.GroupNorm(num_groups_1, branch_hidden_dims[0]))
            branch_layers.append(nn.ReLU())
            branch_layers.append(nn.Dropout(dropout_rate))
            # Second hidden layer of the branch
            branch_layers.append(nn.Linear(branch_hidden_dims[0], branch_hidden_dims[1]))
            num_groups_2 = min(32, branch_hidden_dims[1] // 4)
            if num_groups_2 == 0:
                num_groups_2 = 1
            branch_layers.append(nn.GroupNorm(num_groups_2, branch_hidden_dims[1]))
            branch_layers.append(nn.ReLU())
            branch_layers.append(nn.Dropout(dropout_rate))
            # Embedding layer for the branch
            branch_layers.append(nn.Linear(branch_hidden_dims[1], branch_embedding_dim))
            num_groups_3 = min(32, branch_embedding_dim // 4)
            if num_groups_3 == 0:
                num_groups_3 = 1
            branch_layers.append(nn.GroupNorm(num_groups_3, branch_embedding_dim))
            branch_layers.append(nn.ReLU())
            # No dropout after the final branch embedding
            self.branches[fp_name] = nn.Sequential(*branch_layers)

        concatenated_dim = len(self.fp_list) * branch_embedding_dim
        common_layers_list = []
        # First common hidden layer
        common_layers_list.append(nn.Linear(concatenated_dim, common_hidden_dims[0]))
        num_groups_4 = min(32, common_hidden_dims[0] // 4)
        if num_groups_4 == 0:
            num_groups_4 = 1
        common_layers_list.append(nn.GroupNorm(num_groups_4, common_hidden_dims[0]))
        common_layers_list.append(nn.ReLU())
        common_layers_list.append(nn.Dropout(dropout_rate))
        # Second common hidden layer
        common_layers_list.append(nn.Linear(common_hidden_dims[0], common_hidden_dims[1]))
        num_groups_5 = min(32, common_hidden_dims[1] // 4)
        if num_groups_5 == 0:
            num_groups_5 = 1
        common_layers_list.append(nn.GroupNorm(num_groups_5, common_hidden_dims[1]))
        common_layers_list.append(nn.ReLU())
        common_layers_list.append(nn.Dropout(dropout_rate))
        # Output layer
        common_layers_list.append(nn.Linear(common_hidden_dims[1], 1))
        self.common_layers = nn.Sequential(*common_layers_list)

    def forward(self, x):
        branch_outputs = []
        for fp_name in self.fp_list:
            branch_outputs.append(self.branches[fp_name](x[fp_name].float()))
        concatenated = torch.cat(branch_outputs, dim=1)
        return self.common_layers(concatenated)

    def extract_penultimate(self, x, return_logits=True):
        """
        Returns (penultimate_layer_output, logits) for a given input dict x.
        penultimate_layer_output: output of the last hidden layer before the final output layer.
        logits: output of the final layer (before sigmoid).
        """
        branch_outputs = []
        for fp_name in self.fp_list:
            branch_outputs.append(self.branches[fp_name](x[fp_name].float()))
        concatenated = torch.cat(branch_outputs, dim=1)
        # Forward through all but the last (output) layer of common_layers
        out = concatenated
        for layer in list(self.common_layers)[:-1]:
            out = layer(out)
        penultimate = out
        logits = self.common_layers[-1](penultimate)
        if return_logits:
            return penultimate, logits
        else:
            return penultimate

    def get_penultimate_dim(self):
        """
        Returns the output dimension of the last common hidden layer (penultimate layer).
        """
        return self.common_hidden_dims[-1]
