import torch
from typing import Dict, Any
import re

def create_semantic_key_mapping(seva_keys_file: str = "/home/ruofanl/Projects/stable-virtual-camera/seva_keys.txt",
                               savediff_keys_file: str = "/home/ruofanl/Projects/stable-virtual-camera/save_diff_keys.txt") -> Dict[str, str]:
    """
    Create a semantic mapping between SEVA and SaveDiff keys based on model architecture.
    """

    # Read the key files
    with open(seva_keys_file, 'r') as f:
        seva_keys = [line.strip() for line in f.readlines() if line.strip()]

    with open(savediff_keys_file, 'r') as f:
        savediff_keys = [line.strip() for line in f.readlines() if line.strip()]

    if len(seva_keys) != len(savediff_keys):
        raise ValueError(f"Key count mismatch: SEVA has {len(seva_keys)} keys, SaveDiff has {len(savediff_keys)} keys")

    mapping = {}

    def parse_seva_key(key):
        """Parse SEVA key to extract semantic components."""
        if key.startswith('module.'):
            key = key[7:]  # Remove 'module.' prefix

        # Time embedding
        if key.startswith('time_embed.'):
            layer_num = key.split('.')[1]
            param_type = key.split('.')[-1]
            return ('time_embed', layer_num, param_type)

        # Input blocks (down blocks)
        elif key.startswith('input_blocks.'):
            parts = key.split('.')
            block_idx = int(parts[1])
            sub_idx = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
            remaining = '.'.join(parts[3:]) if len(parts) > 3 else parts[2]
            return ('input_blocks', block_idx, sub_idx, remaining)

        # Output blocks (up blocks)
        elif key.startswith('output_blocks.'):
            parts = key.split('.')
            block_idx = int(parts[1])
            sub_idx = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
            remaining = '.'.join(parts[3:]) if len(parts) > 3 else parts[2]
            return ('output_blocks', block_idx, sub_idx, remaining)

        # Middle block
        elif key.startswith('middle_block.'):
            parts = key.split('.')
            sub_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            remaining = '.'.join(parts[2:]) if len(parts) > 2 else parts[1]
            return ('middle_block', sub_idx, remaining)

        # Output layers
        elif key.startswith('out.'):
            parts = key.split('.')
            layer_idx = int(parts[1])
            param_type = parts[2]
            return ('out', layer_idx, param_type)

        return ('unknown', key)

    def parse_savediff_key(key):
        """Parse SaveDiff key to extract semantic components."""
        # Conv input
        if key.startswith('conv_in.'):
            param_type = key.split('.')[-1]
            return ('conv_in', param_type)

        # Time embedding
        elif key.startswith('time_embedding.'):
            parts = key.split('.')
            layer_name = parts[1]  # linear_1 or linear_2
            param_type = parts[2]
            return ('time_embedding', layer_name, param_type)

        # Down blocks
        elif key.startswith('down_blocks.'):
            parts = key.split('.')
            block_idx = int(parts[1])
            remaining = '.'.join(parts[2:])
            return ('down_blocks', block_idx, remaining)

        # Up blocks
        elif key.startswith('up_blocks.'):
            parts = key.split('.')
            block_idx = int(parts[1])
            remaining = '.'.join(parts[2:])
            return ('up_blocks', block_idx, remaining)

        # Mid block
        elif key.startswith('mid_block.'):
            remaining = '.'.join(key.split('.')[1:])
            return ('mid_block', remaining)

        # Output layers
        elif key.startswith('conv_norm_out.') or key.startswith('conv_out.'):
            parts = key.split('.')
            layer_name = parts[0]  # conv_norm_out or conv_out
            param_type = parts[1]
            return ('output', layer_name, param_type)

        return ('unknown', key)

    # Create semantic groups for both key sets
    seva_groups = {}
    savediff_groups = {}

    for key in seva_keys:
        parsed = parse_seva_key(key)
        if parsed[0] not in seva_groups:
            seva_groups[parsed[0]] = []
        seva_groups[parsed[0]].append((key, parsed))

    for key in savediff_keys:
        parsed = parse_savediff_key(key)
        if parsed[0] not in savediff_groups:
            savediff_groups[parsed[0]] = []
        savediff_groups[parsed[0]].append((key, parsed))

    # Create mappings for each semantic group

    # Time embedding mapping
    if 'time_embed' in seva_groups and 'time_embedding' in savediff_groups:
        seva_time = sorted(seva_groups['time_embed'], key=lambda x: (x[1][1], x[1][2]))
        savediff_time = sorted(savediff_groups['time_embedding'], key=lambda x: (x[1][1], x[1][2]))

        # Map based on layer order: 0->linear_1, 2->linear_2
        time_map = {'0': 'linear_1', '2': 'linear_2'}
        for seva_key, seva_parsed in seva_time:
            layer_num = seva_parsed[1]
            param_type = seva_parsed[2]
            if layer_num in time_map:
                target_layer = time_map[layer_num]
                # Find matching savediff key
                for savediff_key, savediff_parsed in savediff_time:
                    if savediff_parsed[1] == target_layer and savediff_parsed[2] == param_type:
                        mapping[seva_key] = savediff_key
                        break

    # Input blocks -> Down blocks mapping
    if 'input_blocks' in seva_groups and 'down_blocks' in savediff_groups:
        seva_input = seva_groups['input_blocks']
        savediff_down = savediff_groups['down_blocks']

        # Group by block index
        seva_by_block = {}
        for seva_key, seva_parsed in seva_input:
            block_idx = seva_parsed[1]
            if block_idx not in seva_by_block:
                seva_by_block[block_idx] = []
            seva_by_block[block_idx].append((seva_key, seva_parsed))

        savediff_by_block = {}
        for savediff_key, savediff_parsed in savediff_down:
            block_idx = savediff_parsed[1]
            if block_idx not in savediff_by_block:
                savediff_by_block[block_idx] = []
            savediff_by_block[block_idx].append((savediff_key, savediff_parsed))

        # Map input_blocks.0.0 -> conv_in
        if 0 in seva_by_block and 'conv_in' in savediff_groups:
            for seva_key, seva_parsed in seva_by_block[0]:
                if seva_parsed[2] == 0:  # sub_idx == 0
                    param_type = seva_parsed[3]
                    for savediff_key, savediff_parsed in savediff_groups['conv_in']:
                        if savediff_parsed[1] == param_type:
                            mapping[seva_key] = savediff_key
                            break

        # Map remaining input blocks to down blocks (input_blocks.N -> down_blocks.N-1)
        for seva_block_idx in seva_by_block:
            if seva_block_idx == 0:
                continue  # Skip block 0, already handled as conv_in

            # Handle downsampler blocks (input_blocks with single sub-component)
            if seva_block_idx in [3, 6, 9]:  # Downsample blocks
                down_block_idx = (seva_block_idx - 1) // 3  # Map to corresponding down block
                if down_block_idx in savediff_by_block:
                    # Map downsampler
                    for seva_key, seva_parsed in seva_by_block[seva_block_idx]:
                        if 'op.' in seva_parsed[3]:  # Downsampler op
                            param_type = seva_parsed[3].split('.')[-1]
                            # Find corresponding downsamplers.0.conv in savediff
                            for savediff_key, savediff_parsed in savediff_by_block[down_block_idx]:
                                if 'downsamplers.0.conv.' + param_type in savediff_parsed[2]:
                                    mapping[seva_key] = savediff_key
                                    break
            else:
                # Regular blocks with resnets and attentions
                down_block_idx = max(0, (seva_block_idx - 1) // 3)
                if down_block_idx in savediff_by_block:
                    # Map by component similarity
                    for seva_key, seva_parsed in seva_by_block[seva_block_idx]:
                        remaining = seva_parsed[3]
                        # Find best match in savediff block
                        best_match = None
                        max_similarity = 0

                        for savediff_key, savediff_parsed in savediff_by_block[down_block_idx]:
                            savediff_remaining = savediff_parsed[2]
                            # Calculate similarity based on common substrings
                            similarity = calculate_key_similarity(remaining, savediff_remaining)
                            if similarity > max_similarity:
                                max_similarity = similarity
                                best_match = savediff_key

                        if best_match and max_similarity > 0.7:  # Threshold for similarity
                            mapping[seva_key] = best_match

    # Handle remaining mappings by similarity matching
    mapped_seva_keys = set(mapping.keys())
    mapped_savediff_keys = set(mapping.values())

    unmapped_seva = [k for k in seva_keys if k not in mapped_seva_keys]
    unmapped_savediff = [k for k in savediff_keys if k not in mapped_savediff_keys]

    # Match remaining keys by similarity
    for seva_key in unmapped_seva:
        best_match = None
        max_similarity = 0

        for savediff_key in unmapped_savediff:
            similarity = calculate_key_similarity(seva_key, savediff_key)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = savediff_key

        if best_match and max_similarity > 0.5:
            mapping[seva_key] = best_match
            unmapped_savediff.remove(best_match)

    return mapping

def calculate_key_similarity(key1: str, key2: str) -> float:
    """Calculate similarity between two parameter keys."""
    # Remove prefixes for comparison
    key1_clean = key1.replace('module.', '')
    key2_clean = key2

    # Split into components
    parts1 = key1_clean.split('.')
    parts2 = key2_clean.split('.')

    # Check for common patterns
    score = 0.0

    # Weight and bias matching
    if parts1[-1] == parts2[-1] and parts1[-1] in ['weight', 'bias']:
        score += 0.3

    # Component matching
    common_components = ['in_layers', 'out_layers', 'emb_layers', 'dense_emb_layers',
                        'norm', 'proj_in', 'proj_out', 'attn1', 'attn2', 'ff', 'to_q',
                        'to_k', 'to_v', 'to_out', 'norm1', 'norm2', 'norm3', 'conv']

    for component in common_components:
        if component in key1_clean and component in key2_clean:
            score += 0.1

    # Transformer block patterns
    if 'transformer_blocks' in key1_clean and 'transformer_blocks' in key2_clean:
        score += 0.2

    if 'time_mix_blocks' in key1_clean and 'time_mix_blocks' in key2_clean:
        score += 0.2

    # Attention patterns
    if any(attn in key1_clean for attn in ['attn1', 'attn2']) and any(attn in key2_clean for attn in ['attn1', 'attn2']):
        score += 0.1

    return min(score, 1.0)

def convert_seva_to_savediff_state_dict(seva_state_dict: Dict[str, torch.Tensor],
                                       seva_keys_file: str = "/home/ruofanl/Projects/stable-virtual-camera/seva_keys.txt",
                                       savediff_keys_file: str = "/home/ruofanl/Projects/stable-virtual-camera/save_diff_keys.txt") -> Dict[str, torch.Tensor]:
    """
    Convert a state_dict from SEVA format to SaveDiff format using semantic key mapping.

    Args:
        seva_state_dict (Dict[str, torch.Tensor]): The input state_dict with SEVA keys
        seva_keys_file (str): Path to file containing SEVA key names
        savediff_keys_file (str): Path to file containing SaveDiff key names

    Returns:
        Dict[str, torch.Tensor]: The converted state_dict with SaveDiff keys
    """

    # Create semantic mapping
    key_mapping = create_semantic_key_mapping(seva_keys_file, savediff_keys_file)

    # Convert the state_dict
    converted_state_dict = {}

    for seva_key, tensor in seva_state_dict.items():
        if seva_key in key_mapping:
            savediff_key = key_mapping[seva_key]
            converted_state_dict[savediff_key] = tensor
        else:
            # If key not found in mapping, keep original key and warn
            print(f"Warning: Key '{seva_key}' not found in mapping. Keeping original key.")
            converted_state_dict[seva_key] = tensor

    # Verify all expected keys are present
    with open(savediff_keys_file, 'r') as f:
        expected_keys = set(line.strip() for line in f.readlines() if line.strip())

    missing_keys = expected_keys - set(converted_state_dict.keys())
    if missing_keys:
        print(f"Warning: {len(missing_keys)} expected keys are missing in the converted state_dict")
        print("First 10 missing keys:", list(missing_keys)[:10])

    return converted_state_dict


def convert_savediff_to_seva_state_dict(savediff_state_dict: Dict[str, torch.Tensor],
                                       seva_keys_file: str = "/home/ruofanl/Projects/stable-virtual-camera/seva_keys.txt",
                                       savediff_keys_file: str = "/home/ruofanl/Projects/stable-virtual-camera/save_diff_keys.txt") -> Dict[str, torch.Tensor]:
    """
    Convert a state_dict from SaveDiff format to SEVA format using key mapping files.

    Args:
        savediff_state_dict (Dict[str, torch.Tensor]): The input state_dict with SaveDiff keys
        seva_keys_file (str): Path to file containing SEVA key names
        savediff_keys_file (str): Path to file containing SaveDiff key names

    Returns:
        Dict[str, torch.Tensor]: The converted state_dict with SEVA keys
    """

    # Read the key mapping files
    with open(seva_keys_file, 'r') as f:
        seva_keys = [line.strip() for line in f.readlines() if line.strip()]

    with open(savediff_keys_file, 'r') as f:
        savediff_keys = [line.strip() for line in f.readlines() if line.strip()]

    # Verify both files have the same number of keys
    if len(seva_keys) != len(savediff_keys):
        raise ValueError(f"Key count mismatch: SEVA has {len(seva_keys)} keys, SaveDiff has {len(savediff_keys)} keys")

    # Create reverse mapping dictionary (savediff -> seva)
    key_mapping = dict(zip(savediff_keys, seva_keys))

    # Convert the state_dict
    converted_state_dict = {}

    for savediff_key, tensor in savediff_state_dict.items():
        if savediff_key in key_mapping:
            seva_key = key_mapping[savediff_key]
            converted_state_dict[seva_key] = tensor
        else:
            # If key not found in mapping, keep original key and warn
            print(f"Warning: Key '{savediff_key}' not found in mapping. Keeping original key.")
            converted_state_dict[savediff_key] = tensor

    # Verify all expected keys are present
    missing_keys = set(seva_keys) - set(converted_state_dict.keys())
    if missing_keys:
        print(f"Warning: {len(missing_keys)} expected keys are missing in the converted state_dict")
        print("First 10 missing keys:", list(missing_keys)[:10])

    return converted_state_dict


def get_key_mapping(seva_keys_file: str = "/home/ruofanl/Projects/stable-virtual-camera/seva_keys.txt",
                   savediff_keys_file: str = "/home/ruofanl/Projects/stable-virtual-camera/save_diff_keys.txt") -> Dict[str, str]:
    """
    Get the key mapping dictionary from SEVA to SaveDiff format.

    Args:
        seva_keys_file (str): Path to file containing SEVA key names
        savediff_keys_file (str): Path to file containing SaveDiff key names

    Returns:
        Dict[str, str]: Mapping from SEVA keys to SaveDiff keys
    """

    # Read the key mapping files
    with open(seva_keys_file, 'r') as f:
        seva_keys = [line.strip() for line in f.readlines() if line.strip()]

    with open(savediff_keys_file, 'r') as f:
        savediff_keys = [line.strip() for line in f.readlines() if line.strip()]

    # Verify both files have the same number of keys
    if len(seva_keys) != len(savediff_keys):
        raise ValueError(f"Key count mismatch: SEVA has {len(seva_keys)} keys, SaveDiff has {len(savediff_keys)} keys")

    return dict(zip(seva_keys, savediff_keys))


# Example usage
if __name__ == "__main__":
    # Example of how to use the converter

    # Load a SEVA model state_dict (example)
    # seva_state_dict = torch.load('seva_model.pt')

    # Convert to SaveDiff format
    # converted_state_dict = convert_seva_to_savediff_state_dict(seva_state_dict)

    # Save the converted state_dict
    # torch.save(converted_state_dict, 'converted_model.pt')

    # Print the key mapping for inspection
    mapping = get_key_mapping()
    print(f"Total parameter mappings: {len(mapping)}")
    print("\nFirst 10 mappings:")
    for i, (seva_key, savediff_key) in enumerate(mapping.items()):
        if i >= 10:
            break
        print(f"{seva_key} -> {savediff_key}")
