import torch
import torch.nn.functional as F
import sys
sys.path.append('/home/ruofanl/Projects/video-to-video')

from seva.model import SGMWrapper
from seva.utils import load_model
from src.models.custom_unet_mv import UNetMV2DConditionModel

def hook_layer_outputs(module, name, storage):
    """Hook function to capture layer outputs"""
    def hook(module, input, output):
        if isinstance(output, tuple):
            storage[name] = output[0] if len(output) > 0 else output
        else:
            storage[name] = output
    return hook

def detailed_layer_comparison():
    # Load models
    MODEL = SGMWrapper(
        load_model(
            model_version=1.1,
            pretrained_model_name_or_path="stabilityai/stable-virtual-camera",
            weight_name="model.safetensors",
            device="cpu",
            verbose=True,
        ).eval()
    )

    unet = UNetMV2DConditionModel.from_pretrained('/home/ruofanl/Projects/exp_outputs/SEVA/unet', subfolder="unet")

    # Create test inputs (use same seed for reproducibility)
    torch.manual_seed(42)
    x = torch.randn(2, 11, 64, 64)
    t = torch.ones(1)
    d = torch.randn(2, 6, 64, 64)
    c = torch.zeros(2, 1, 1024)

    # Storage for intermediate outputs
    seva_outputs = {}
    unet_outputs = {}

    # Register hooks for SEVA model
    seva_hooks = []
    for i, block in enumerate(MODEL.module.input_blocks):
        hook = block.register_forward_hook(hook_layer_outputs(block, f"input_block_{i}", seva_outputs))
        seva_hooks.append(hook)

    middle_hook = MODEL.module.middle_block.register_forward_hook(hook_layer_outputs(MODEL.module.middle_block, "middle_block", seva_outputs))
    seva_hooks.append(middle_hook)

    for i, block in enumerate(MODEL.module.output_blocks):
        hook = block.register_forward_hook(hook_layer_outputs(block, f"output_block_{i}", seva_outputs))
        seva_hooks.append(hook)

    # Register hooks for UNet model
    unet_hooks = []
    for i, block in enumerate(unet.down_blocks):
        hook = block.register_forward_hook(hook_layer_outputs(block, f"down_block_{i}", unet_outputs))
        unet_hooks.append(hook)

    mid_hook = unet.mid_block.register_forward_hook(hook_layer_outputs(unet.mid_block, "mid_block", unet_outputs))
    unet_hooks.append(mid_hook)

    for i, block in enumerate(unet.up_blocks):
        hook = block.register_forward_hook(hook_layer_outputs(block, f"up_block_{i}", unet_outputs))
        unet_hooks.append(hook)

    with torch.no_grad():
        print("Running forward passes...")
        y_seva = MODEL.module(x, t, c, d, num_frames=2)
        y_diff = unet(x[None, :], t, c, d[None, :]).sample[0]

        print(f"Final output difference: {torch.max(torch.abs(y_seva - y_diff))}")

        # Compare captured intermediate outputs
        print("\n=== INTERMEDIATE LAYER COMPARISONS ===")

        # Compare down/input blocks
        min_blocks = min(len(MODEL.module.input_blocks), len(unet.down_blocks))
        for i in range(min_blocks):
            seva_key = f"input_block_{i}"
            unet_key = f"down_block_{i}"

            if seva_key in seva_outputs and unet_key in unet_outputs:
                seva_out = seva_outputs[seva_key]
                unet_out = unet_outputs[unet_key]

                # UNet returns (hidden_states, res_samples)
                if isinstance(unet_out, tuple):
                    unet_out = unet_out[0]  # Take hidden states

                if seva_out.shape == unet_out.shape:
                    diff = torch.max(torch.abs(seva_out - unet_out))
                    print(f"Block {i}: SEVA {seva_out.shape} vs UNet {unet_out.shape}, max diff: {diff}")
                else:
                    print(f"Block {i}: Shape mismatch - SEVA {seva_out.shape} vs UNet {unet_out.shape}")

        # Compare middle blocks
        if "middle_block" in seva_outputs and "mid_block" in unet_outputs:
            seva_mid = seva_outputs["middle_block"]
            unet_mid = unet_outputs["mid_block"]

            if seva_mid.shape == unet_mid.shape:
                mid_diff = torch.max(torch.abs(seva_mid - unet_mid))
                print(f"Middle block: SEVA {seva_mid.shape} vs UNet {unet_mid.shape}, max diff: {mid_diff}")
            else:
                print(f"Middle block: Shape mismatch - SEVA {seva_mid.shape} vs UNet {unet_mid.shape}")

        # Compare up/output blocks
        min_up_blocks = min(len(MODEL.module.output_blocks), len(unet.up_blocks))
        for i in range(min_up_blocks):
            seva_key = f"output_block_{i}"
            unet_key = f"up_block_{i}"

            if seva_key in seva_outputs and unet_key in unet_outputs:
                seva_out = seva_outputs[seva_key]
                unet_out = unet_outputs[unet_key]

                if seva_out.shape == unet_out.shape:
                    diff = torch.max(torch.abs(seva_out - unet_out))
                    print(f"Up block {i}: SEVA {seva_out.shape} vs UNet {unet_out.shape}, max diff: {diff}")
                else:
                    print(f"Up block {i}: Shape mismatch - SEVA {seva_out.shape} vs UNet {unet_out.shape}")

    # Clean up hooks
    for hook in seva_hooks + unet_hooks + [middle_hook, mid_hook]:
        hook.remove()

    return y_seva, y_diff

if __name__ == "__main__":
    detailed_layer_comparison()
