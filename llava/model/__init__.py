AVAILABLE_MODELS = {
    "video_salmonn_2": "VideoSALMONN2ForCausalLM, VideoSALMONN2Config",
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except Exception as e:
        raise e
