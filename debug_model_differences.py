import torch
import torch.nn.functional as F
import sys
sys.path.append('/home/ruofanl/Projects/video-to-video')

from seva.model import SGMWrapper
from seva.utils import load_model
from src.models.custom_unet_mv import UNetMV2DConditionModel

def compare_time_embeddings(model_seva, model_unet, t, x_shape):
    """Compare time embedding generation between models"""
    print("=== TIME EMBEDDING COMPARISON ===")

    # SEVA time embedding
    from seva.modules.layers import timestep_embedding
    seva_t_emb = timestep_embedding(t, model_seva.model_channels)
    seva_t_emb_final = model_seva.time_embed(seva_t_emb)
    print(f"SEVA t_emb shape: {seva_t_emb.shape}, final: {seva_t_emb_final.shape}")
    print(f"SEVA t_emb[:5]: {seva_t_emb.flatten()[:5]}")
    print(f"SEVA t_emb_final[:5]: {seva_t_emb_final.flatten()[:5]}")

    # UNet time embedding
    batch_size, num_frames = x_shape[:2]
    timesteps = t.expand(batch_size)
    unet_t_emb = model_unet.time_proj(timesteps)
    unet_t_emb = unet_t_emb.to(dtype=torch.float32)
    unet_emb = model_unet.time_embedding(unet_t_emb)
    unet_emb_repeated = unet_emb.repeat_interleave(num_frames, dim=0)

    print(f"UNet timesteps: {timesteps}")
    print(f"UNet t_emb shape: {unet_t_emb.shape}, final: {unet_emb.shape}, repeated: {unet_emb_repeated.shape}")
    print(f"UNet t_emb[:5]: {unet_t_emb.flatten()[:5]}")
    print(f"UNet emb[:5]: {unet_emb.flatten()[:5]}")
    print(f"UNet emb_repeated[:5]: {unet_emb_repeated.flatten()[:5]}")

    return seva_t_emb_final, unet_emb_repeated

def compare_input_processing(model_seva, model_unet, x_seva, x_unet, dense_emb_seva, dense_emb_unet):
    """Compare initial input processing"""
    print("\n=== INPUT PROCESSING COMPARISON ===")

    # SEVA input processing (through first conv)
    seva_input_conv = model_seva.input_blocks[0][0]  # First conv layer
    seva_h = seva_input_conv(x_seva)
    print(f"SEVA input conv output shape: {seva_h.shape}")
    print(f"SEVA input conv output[:5]: {seva_h.flatten()[:5]}")

    # UNet input processing
    x_unet_flat = x_unet.flatten(0, 1)
    dense_emb_unet_flat = dense_emb_unet.flatten(0, 1)
    unet_h = model_unet.conv_in(x_unet_flat)
    print(f"UNet input conv output shape: {unet_h.shape}")
    print(f"UNet input conv output[:5]: {unet_h.flatten()[:5]}")

    return seva_h, unet_h, dense_emb_unet_flat

def debug_models():
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

    # Create test inputs
    x = torch.randn(2, 11, 64, 64)
    t = torch.ones(1)
    d = torch.randn(2, 6, 64, 64)
    c = torch.zeros(2, 1, 1024)

    print(f"Input shapes - x: {x.shape}, t: {t.shape}, d: {d.shape}, c: {c.shape}")

    with torch.no_grad():
        # Compare time embeddings
        seva_t_emb, unet_t_emb = compare_time_embeddings(
            MODEL.module, unet, t, x[None, :].shape
        )

        # Compare input processing
        seva_h, unet_h, dense_unet_flat = compare_input_processing(
            MODEL.module, unet, x, x[None, :], d, d[None, :]
        )

        print(f"\nTime embedding difference (should be 0 if same):")
        print(f"Max diff: {torch.max(torch.abs(seva_t_emb - unet_t_emb))}")

        print(f"\nInput processing difference (should be 0 if same):")
        print(f"Max diff: {torch.max(torch.abs(seva_h - unet_h))}")

        # Check if conv weights are the same
        print(f"\nConv weight comparison:")
        seva_conv_weight = MODEL.module.input_blocks[0][0].weight
        unet_conv_weight = unet.conv_in.weight
        print(f"SEVA conv weight shape: {seva_conv_weight.shape}")
        print(f"UNet conv weight shape: {unet_conv_weight.shape}")
        print(f"Conv weight diff: {torch.max(torch.abs(seva_conv_weight - unet_conv_weight))}")

        # Full forward pass comparison
        print(f"\n=== FULL FORWARD PASS ===")
        y_seva = MODEL.module(x, t, c, d, num_frames=2)
        y_diff = unet(x[None, :], t, c, d[None, :]).sample[0]

        print(f"SEVA output shape: {y_seva.shape}")
        print(f"UNet output shape: {y_diff.shape}")
        print(f"Output difference: {torch.max(torch.abs(y_seva - y_diff))}")
        print(f"Are outputs close? {torch.allclose(y_seva, y_diff, atol=1e-3)}")

if __name__ == "__main__":
    debug_models()
