import torch
import torch.nn as nn

def inverse_sigmoid(x, eps=1e-5):
    """
    Inverse function of sigmoid.
    
    Args:
        x (Tensor): Input tensor.
        eps (float): Small value to avoid division by zero.
    
    Returns:
        Tensor: The inverse sigmoid of x.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class DetectionTransformerDecoder(nn.Module):
    """
    Detection Transformer Decoder in pure PyTorch.
    
    Args:
        layers (list[nn.Module]): A list of decoder layer modules.
        return_intermediate (bool): Whether to return intermediate outputs.
            If True, the decoder will output a stack of intermediate layer outputs and reference points.
            Otherwise, it returns only the final output.
    """
    def __init__(self, layers, return_intermediate=False):
        super(DetectionTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self, query, *args, reference_points=None, reg_branches=None, key_padding_mask=None, **kwargs):
        """
        Forward function for the DetectionTransformerDecoder.
        
        Args:
            query (Tensor): Input query with shape (num_query, bs, embed_dims).
            reference_points (Tensor): Reference points with shape (bs, num_query, num_level, 2) 
                or (bs, num_query, 4) when used in two-stage refinement.
            reg_branches (ModuleList, optional): Modules for refining regression results.
                Only used if available; otherwise, pass as None.
            key_padding_mask (Tensor, optional): Padding mask for key inputs.
            *args, **kwargs: Other additional arguments to be passed to each layer.
        
        Returns:
            If return_intermediate is False:
                Tuple[Tensor, Tensor]: 
                    - output: (num_query, bs, embed_dims)
                    - reference_points: Updated reference points.
            If return_intermediate is True:
                Tuple[Tensor, Tensor]:
                    - stacked intermediate outputs: shape (num_layers, num_query, bs, embed_dims)
                    - stacked reference points: shape (num_layers, bs, num_query, ...)
        """
        output = query  # (num_query, bs, embed_dims)
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            # Prepare reference points input:
            # Expected shape transformation: (bs, num_query, num_level, 2) -> add a dimension: (bs, num_query, 1, num_level, 2)
            # (참고: 각 layer에서는 해당 형식에 맞게 처리하도록 layer 내 구현이 필요합니다.)
            reference_points_input = reference_points[..., :2].unsqueeze(2)  # (bs, num_query, 1, num_level, 2)
            
            # Pass through the current decoder layer.
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            
            # Permute output to (bs, num_query, embed_dims) for further processing.
            output = output.permute(1, 0, 2)
            
            # If regression branch(es) are provided, update reference points.
            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                # tmp should be of a shape that allows updating reference_points.
                # 여기서는 reference_points의 마지막 차원이 3이어야 한다고 가정합니다.
                assert reference_points.shape[-1] == 3, "Expected reference_points last dim to be 3 for regression refinement."
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            
            # Permute back to (num_query, bs, embed_dims)
            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points
