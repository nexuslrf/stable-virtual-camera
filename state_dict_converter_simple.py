import torch
from typing import Dict, Any

def create_direct_key_mapping(seva_keys_file: str = "/home/ruofanl/Projects/stable-virtual-camera/seva_keys.txt",
                             savediff_keys_file: str = "/home/ruofanl/Projects/stable-virtual-camera/save_diff_keys.txt") -> Dict[str, str]:
    """
    Create a direct mapping between SEVA and SaveDiff keys based on architectural understanding.

    Based on the model structures:
    - SEVA uses module.* prefix, SaveDiff doesn't
    - module.time_embed.0 -> time_embedding.linear_1
    - module.time_embed.2 -> time_embedding.linear_2
    - module.input_blocks.0.0 -> conv_in
    - module.input_blocks.{1,2}.{0,1} -> down_blocks.0.{resnets,attentions}.{0,1}
    - module.input_blocks.{4,5}.{0,1} -> down_blocks.1.{resnets,attentions}.{0,1}
    - module.input_blocks.{7,8}.{0,1} -> down_blocks.2.{resnets,attentions}.{0,1}
    - module.input_blocks.{10,11}.0 -> down_blocks.3.resnets.{0,1}
    - module.input_blocks.{3,6,9}.0.op -> down_blocks.{0,1,2}.downsamplers.0.conv
    - module.middle_block -> mid_block
    - module.output_blocks -> up_blocks
    - module.out.0 -> conv_norm_out
    - module.out.2 -> conv_out
    """

    # Read the key files
    with open(seva_keys_file, 'r') as f:
        seva_keys = [line.strip() for line in f.readlines() if line.strip()]

    with open(savediff_keys_file, 'r') as f:
        savediff_keys = [line.strip() for line in f.readlines() if line.strip()]

    if len(seva_keys) != len(savediff_keys):
        raise ValueError(f"Key count mismatch: SEVA has {len(seva_keys)} keys, SaveDiff has {len(savediff_keys)} keys")

    # Create organized mappings
    seva_organized = organize_seva_keys(seva_keys)
    savediff_organized = organize_savediff_keys(savediff_keys)

    mapping = {}

    # Time embedding mapping
    for seva_key in seva_organized.get('time_embed', []):
        if '.0.' in seva_key:
            param_type = seva_key.split('.')[-1]
            target_key = f"time_embedding.linear_1.{param_type}"
            if target_key in savediff_keys:
                mapping[seva_key] = target_key
        elif '.2.' in seva_key:
            param_type = seva_key.split('.')[-1]
            target_key = f"time_embedding.linear_2.{param_type}"
            if target_key in savediff_keys:
                mapping[seva_key] = target_key

    # Conv input mapping (input_blocks.0.0 -> conv_in)
    for seva_key in seva_organized.get('input_blocks', []):
        if seva_key.startswith('module.input_blocks.0.0.'):
            param_type = seva_key.split('.')[-1]
            target_key = f"conv_in.{param_type}"
            if target_key in savediff_keys:
                mapping[seva_key] = target_key

    # Down blocks mapping
    down_block_mappings = {
        # input_blocks.{1,2} -> down_blocks.0
        (1, 0): ('down_blocks.0.resnets.0', 'resnet'),
        (1, 1): ('down_blocks.0.attentions.0', 'attention'),
        (2, 0): ('down_blocks.0.resnets.1', 'resnet'),
        (2, 1): ('down_blocks.0.attentions.1', 'attention'),
        # input_blocks.3 -> down_blocks.0.downsamplers.0
        (3, 0): ('down_blocks.0.downsamplers.0', 'downsampler'),

        # input_blocks.{4,5} -> down_blocks.1
        (4, 0): ('down_blocks.1.resnets.0', 'resnet'),
        (4, 1): ('down_blocks.1.attentions.0', 'attention'),
        (5, 0): ('down_blocks.1.resnets.1', 'resnet'),
        (5, 1): ('down_blocks.1.attentions.1', 'attention'),
        # input_blocks.6 -> down_blocks.1.downsamplers.0
        (6, 0): ('down_blocks.1.downsamplers.0', 'downsampler'),

        # input_blocks.{7,8} -> down_blocks.2
        (7, 0): ('down_blocks.2.resnets.0', 'resnet'),
        (7, 1): ('down_blocks.2.attentions.0', 'attention'),
        (8, 0): ('down_blocks.2.resnets.1', 'resnet'),
        (8, 1): ('down_blocks.2.attentions.1', 'attention'),
        # input_blocks.9 -> down_blocks.2.downsamplers.0
        (9, 0): ('down_blocks.2.downsamplers.0', 'downsampler'),

        # input_blocks.{10,11} -> down_blocks.3
        (10, 0): ('down_blocks.3.resnets.0', 'resnet'),
        (11, 0): ('down_blocks.3.resnets.1', 'resnet'),
    }

    for seva_key in seva_organized.get('input_blocks', []):
        parts = seva_key.replace('module.', '').split('.')
        if len(parts) >= 3:
            block_idx = int(parts[1])
            sub_idx = int(parts[2]) if parts[2].isdigit() else None

            if (block_idx, sub_idx) in down_block_mappings:
                target_prefix, block_type = down_block_mappings[(block_idx, sub_idx)]
                remaining_path = '.'.join(parts[3:])

                if block_type == 'downsampler' and 'op.' in remaining_path:
                    # Handle downsampler: op.weight -> conv.weight
                    remaining_path = remaining_path.replace('op.', 'conv.')

                target_key = f"{target_prefix}.{remaining_path}"
                if target_key in savediff_keys:
                    mapping[seva_key] = target_key

    # Middle block mapping
    for seva_key in seva_organized.get('middle_block', []):
        remaining = seva_key.replace('module.middle_block.', '')
        target_key = f"mid_block.{remaining}"

        # Handle specific mappings for middle block structure
        # middle_block.0 -> mid_block.resnets.0
        # middle_block.1 -> mid_block.attentions.0
        # middle_block.2 -> mid_block.resnets.1
        if remaining.startswith('0.'):
            target_key = f"mid_block.resnets.0.{remaining[2:]}"
        elif remaining.startswith('1.'):
            target_key = f"mid_block.attentions.0.{remaining[2:]}"
        elif remaining.startswith('2.'):
            target_key = f"mid_block.resnets.1.{remaining[2:]}"

        if target_key in savediff_keys:
            mapping[seva_key] = target_key

    # Output blocks mapping (similar to input blocks but reversed)
    up_block_mappings = {
        # output_blocks.{0,1,2} -> up_blocks.0
        (0, 0): ('up_blocks.0.resnets.0', 'resnet'),
        (1, 0): ('up_blocks.0.resnets.1', 'resnet'),
        (2, 0): ('up_blocks.0.resnets.2', 'resnet'),
        (2, 1): ('up_blocks.0.upsamplers.0', 'upsampler'),

        # output_blocks.{3,4,5} -> up_blocks.1
        (3, 0): ('up_blocks.1.resnets.0', 'resnet'),
        (3, 1): ('up_blocks.1.attentions.0', 'attention'),
        (4, 0): ('up_blocks.1.resnets.1', 'resnet'),
        (4, 1): ('up_blocks.1.attentions.1', 'attention'),
        (5, 0): ('up_blocks.1.resnets.2', 'resnet'),
        (5, 1): ('up_blocks.1.attentions.2', 'attention'),
        (5, 2): ('up_blocks.1.upsamplers.0', 'upsampler'),

        # output_blocks.{6,7,8} -> up_blocks.2
        (6, 0): ('up_blocks.2.resnets.0', 'resnet'),
        (6, 1): ('up_blocks.2.attentions.0', 'attention'),
        (7, 0): ('up_blocks.2.resnets.1', 'resnet'),
        (7, 1): ('up_blocks.2.attentions.1', 'attention'),
        (8, 0): ('up_blocks.2.resnets.2', 'resnet'),
        (8, 1): ('up_blocks.2.attentions.2', 'attention'),
        (8, 2): ('up_blocks.2.upsamplers.0', 'upsampler'),

        # output_blocks.{9,10,11} -> up_blocks.3
        (9, 0): ('up_blocks.3.resnets.0', 'resnet'),
        (9, 1): ('up_blocks.3.attentions.0', 'attention'),
        (10, 0): ('up_blocks.3.resnets.1', 'resnet'),
        (10, 1): ('up_blocks.3.attentions.1', 'attention'),
        (11, 0): ('up_blocks.3.resnets.2', 'resnet'),
        (11, 1): ('up_blocks.3.attentions.2', 'attention'),
    }

    for seva_key in seva_organized.get('output_blocks', []):
        parts = seva_key.replace('module.', '').split('.')
        if len(parts) >= 3:
            block_idx = int(parts[1])
            sub_idx = int(parts[2]) if parts[2].isdigit() else None

            if (block_idx, sub_idx) in up_block_mappings:
                target_prefix, block_type = up_block_mappings[(block_idx, sub_idx)]
                remaining_path = '.'.join(parts[3:])

                if block_type == 'upsampler' and 'conv.' in remaining_path:
                    # Handle upsampler: already has conv in path
                    pass

                target_key = f"{target_prefix}.{remaining_path}"
                if target_key in savediff_keys:
                    mapping[seva_key] = target_key

    # Output layer mapping
    for seva_key in seva_organized.get('out', []):
        if seva_key == 'module.out.0.weight':
            mapping[seva_key] = 'conv_norm_out.weight'
        elif seva_key == 'module.out.0.bias':
            mapping[seva_key] = 'conv_norm_out.bias'
        elif seva_key == 'module.out.2.weight':
            mapping[seva_key] = 'conv_out.weight'
        elif seva_key == 'module.out.2.bias':
            mapping[seva_key] = 'conv_out.bias'

    # Handle any remaining unmapped keys by fuzzy matching
    mapped_keys = set(mapping.keys())
    unmapped_seva = [k for k in seva_keys if k not in mapped_keys]
    used_savediff = set(mapping.values())
    unused_savediff = [k for k in savediff_keys if k not in used_savediff]

    # Simple fuzzy matching for remaining keys
    for seva_key in unmapped_seva:
        best_match = find_best_match(seva_key, unused_savediff)
        if best_match:
            mapping[seva_key] = best_match
            unused_savediff.remove(best_match)

    return mapping

def organize_seva_keys(seva_keys):
    """Organize SEVA keys by component type."""
    organized = {
        'time_embed': [],
        'input_blocks': [],
        'middle_block': [],
        'output_blocks': [],
        'out': []
    }

    for key in seva_keys:
        if 'time_embed' in key:
            organized['time_embed'].append(key)
        elif 'input_blocks' in key:
            organized['input_blocks'].append(key)
        elif 'middle_block' in key:
            organized['middle_block'].append(key)
        elif 'output_blocks' in key:
            organized['output_blocks'].append(key)
        elif key.startswith('module.out.'):
            organized['out'].append(key)

    return organized

def organize_savediff_keys(savediff_keys):
    """Organize SaveDiff keys by component type."""
    organized = {
        'conv_in': [],
        'time_embedding': [],
        'down_blocks': [],
        'mid_block': [],
        'up_blocks': [],
        'conv_out': []
    }

    for key in savediff_keys:
        if key.startswith('conv_in.'):
            organized['conv_in'].append(key)
        elif key.startswith('time_embedding.'):
            organized['time_embedding'].append(key)
        elif key.startswith('down_blocks.'):
            organized['down_blocks'].append(key)
        elif key.startswith('mid_block.'):
            organized['mid_block'].append(key)
        elif key.startswith('up_blocks.'):
            organized['up_blocks'].append(key)
        elif key.startswith('conv_norm_out.') or key.startswith('conv_out.'):
            organized['conv_out'].append(key)

    return organized

def find_best_match(seva_key, savediff_candidates):
    """Find the best matching SaveDiff key for a SEVA key."""
    seva_clean = seva_key.replace('module.', '')

    best_match = None
    best_score = 0

    for candidate in savediff_candidates:
        score = 0

        # Exact component matches
        common_parts = ['weight', 'bias', 'norm', 'conv', 'proj', 'attn', 'to_q', 'to_k', 'to_v', 'to_out', 'ff', 'net']
        for part in common_parts:
            if part in seva_clean and part in candidate:
                score += 0.2

        # Attention patterns
        if 'attn1' in seva_clean and 'attn1' in candidate:
            score += 0.3
        if 'attn2' in seva_clean and 'attn2' in candidate:
            score += 0.3

        # Transformer patterns
        if 'transformer_blocks' in seva_clean and 'transformer_blocks' in candidate:
            score += 0.2

        if 'time_mix_blocks' in seva_clean and 'time_mix_blocks' in candidate:
            score += 0.2

        # Layer patterns
        if 'in_layers' in seva_clean and 'in_layers' in candidate:
            score += 0.2
        if 'out_layers' in seva_clean and 'out_layers' in candidate:
            score += 0.2
        if 'emb_layers' in seva_clean and 'emb_layers' in candidate:
            score += 0.2
        if 'dense_emb_layers' in seva_clean and 'dense_emb_layers' in candidate:
            score += 0.2

        if score > best_score:
            best_score = score
            best_match = candidate

    return best_match if best_score > 0.4 else None

def convert_seva_to_savediff_state_dict(seva_state_dict: Dict[str, torch.Tensor],
                                       seva_keys_file: str = "/home/ruofanl/Projects/stable-virtual-camera/seva_keys.txt",
                                       savediff_keys_file: str = "/home/ruofanl/Projects/stable-virtual-camera/save_diff_keys.txt") -> Dict[str, torch.Tensor]:
    """
    Convert a state_dict from SEVA format to SaveDiff format.

    Args:
        seva_state_dict (Dict[str, torch.Tensor]): The input state_dict with SEVA keys
        seva_keys_file (str): Path to file containing SEVA key names
        savediff_keys_file (str): Path to file containing SaveDiff key names

    Returns:
        Dict[str, torch.Tensor]: The converted state_dict with SaveDiff keys
    """

    # Create mapping
    key_mapping = create_direct_key_mapping(seva_keys_file, savediff_keys_file)

    # Convert the state_dict
    converted_state_dict = {}

    for seva_key, tensor in seva_state_dict.items():
        if seva_key in key_mapping:
            savediff_key = key_mapping[seva_key]
            converted_state_dict[savediff_key] = tensor
        else:
            print(f"Warning: Key '{seva_key}' not found in mapping. Keeping original key.")
            converted_state_dict[seva_key] = tensor

    # Verify completeness
    with open(savediff_keys_file, 'r') as f:
        expected_keys = set(line.strip() for line in f.readlines() if line.strip())

    missing_keys = expected_keys - set(converted_state_dict.keys())
    if missing_keys:
        print(f"Warning: {len(missing_keys)} expected keys are missing")
        print("First 10 missing keys:", list(missing_keys)[:10])

    extra_keys = set(converted_state_dict.keys()) - expected_keys
    if extra_keys:
        print(f"Warning: {len(extra_keys)} extra keys found")
        print("First 10 extra keys:", list(extra_keys)[:10])

    print(f"Successfully mapped {len(key_mapping)} keys")
    print(f"Conversion complete: {len(converted_state_dict)} parameters in output state_dict")

    return converted_state_dict

def convert_savediff_to_seva_state_dict(savediff_state_dict: Dict[str, torch.Tensor],
                                       seva_keys_file: str = "/home/ruofanl/Projects/stable-virtual-camera/seva_keys.txt",
                                       savediff_keys_file: str = "/home/ruofanl/Projects/stable-virtual-camera/save_diff_keys.txt") -> Dict[str, torch.Tensor]:
    """
    Convert a state_dict from SaveDiff format to SEVA format.
    """

    # Create reverse mapping
    forward_mapping = create_direct_key_mapping(seva_keys_file, savediff_keys_file)
    reverse_mapping = {v: k for k, v in forward_mapping.items()}

    # Convert the state_dict
    converted_state_dict = {}

    for savediff_key, tensor in savediff_state_dict.items():
        if savediff_key in reverse_mapping:
            seva_key = reverse_mapping[savediff_key]
            converted_state_dict[seva_key] = tensor
        else:
            print(f"Warning: Key '{savediff_key}' not found in reverse mapping. Keeping original key.")
            converted_state_dict[savediff_key] = tensor

    return converted_state_dict

# Example usage
if __name__ == "__main__":
    # Test the mapping creation
    mapping = create_direct_key_mapping()
    print(f"Created mapping with {len(mapping)} entries")

    # Show some example mappings
    print("\nExample mappings:")
    for i, (seva_key, savediff_key) in enumerate(mapping.items()):
        if i >= 20:
            break
        print(f"{seva_key:<60} -> {savediff_key}")
