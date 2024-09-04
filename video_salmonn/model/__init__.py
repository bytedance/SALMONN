from .agent import DeepSpeedAgent
from .openllama import OpenLLAMAPEFTModel

def load_model(args):
    agent_name = args['models'][args['model']]['agent_name']
    model_name = args['models'][args['model']]['model_name']
    model = globals()[model_name](**args)
    for name, module in model.named_children():
        if hasattr(module, "gradient_checkpointing_enable"):
            try:
                module.config.use_cache = False
                module.enable_input_require_grads()
                module.gradient_checkpointing_enable()
            except Exception as e:
                print(e)
    agent = globals()[agent_name](model, args)
    return agent
