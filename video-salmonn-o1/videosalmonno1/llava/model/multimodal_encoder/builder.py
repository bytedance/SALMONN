import os
from .clip_encoder import CLIPVisionTower
from .imagebind import ImageBindWrapper
from .siglip_encoder import SigLipVisionTower
from .siglip_encoder_flash import SigLipVisionTowerFlash
from .open_clip_encoder import OpenCLIPVisionTower
from .eva_vit import EvaViTWrapper
from .hf_vision import HFVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    image_processor = getattr(vision_tower_cfg, 'image_processor', getattr(vision_tower_cfg, 'image_processor', "./model_zoo/OpenAI/clip-vit-large-patch14"))
    use_flash_tower = getattr(vision_tower_cfg, 'use_flash_tower', False)

    if vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower in ["imagebind_huge"]:
        return ImageBindWrapper(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "lavis" in vision_tower.lower() or "eva" in vision_tower.lower():
        return EvaViTWrapper(vision_tower, image_processor, args=vision_tower_cfg, **kwargs)
    elif "siglip" in vision_tower:
        if use_flash_tower:
            return SigLipVisionTowerFlash(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
        else:
            return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("hf:"):
        return HFVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("hf-hub"):
        return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
