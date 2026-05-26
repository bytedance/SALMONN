from transformers import Qwen3VLMoeForConditionalGeneration as DefaultQwen3
from qwenvl.model.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
import torch # 建议显式导入torch

def apply_fused_kernel_to_moe():
    print("Applying fused kernel to MoE")
    from qwenvl.model.qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedSparseMoeBlock
    from qwenvl.model import modeling_qwen3_vl_moe

    modeling_qwen3_vl_moe.Qwen3VLMoeTextSparseMoeBlock = Qwen3MoeFusedSparseMoeBlock

apply_fused_kernel_to_moe()

# 1. 加载模型
# default: Load the model on the available device(s)
model_unfused = DefaultQwen3.from_pretrained(
    "/opt/tiger/thu_qwenvl3/output/Qwen3-VL-30B-A3B-Instruct", dtype="auto"
)

model_fused = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    "/opt/tiger/thu_qwenvl3/output/Qwen3-VL-30B-A3B-Instruct", dtype="auto"
)

# 2. 获取 fused 模型的 state_dict，我们将基于它进行修改
# 这是一个好的起点，因为它包含了所有参数的正确键名
fused_state_dict = model_fused.state_dict()

# 3. 遍历 unfused 模型的参数
# 注意：我们遍历 unfused 模型的 state_dict 而不是 named_parameters()，这更直接
for k, v in model_unfused.state_dict().items():
    # 如果参数不是 MoE experts 部分，直接拷贝（如果存在于 fused 模型中）
    if "experts" not in k:
        if k in fused_state_dict:
            fused_state_dict[k] = v
        # 可以加一个 else 语句来捕获那些在 unfused 中存在但在 fused 中不存在的非 expert 参数（用于调试）
        # else:
        #     print(f"Skipping non-expert key not in fused model: {k}")
        continue

    # --- 以下是处理 experts 参数的逻辑，与您的原始逻辑相同 ---
    print(k)
    # 我们只处理那些在 unfused 模型中是 experts，但在 fused 模型中不是的参数
    # `k.replace(".experts", "")` 会将 '...block.mlp.experts.down_proj...' 转换为 '...block.mlp.down_proj...'
    fused_k_base = k.split(".experts.")[0]
    param_name = k.split(".experts.")[1]
    
    if "down_proj" in param_name:
        # 构建 fused 模型的键名，例如 '...mlp.down_proj.weight'
        fused_k = f"{fused_k_base}.down_proj.weight"
        # 更新我们准备好的字典
        # print(fused_state_dict[fused_k].shape, v.shape)
        fused_state_dict[fused_k] = v.transpose(-1, -2)
        
    elif "gate_up_proj" in param_name:
        # 构建 fused 模型的 gate_proj 和 up_proj 的键名
        layer_idx = fused_k_base.split('.')[-1]
        fused_k_gate = f"{fused_k_base}.gate_proj.weight"
        fused_k_up = f"{fused_k_base}.up_proj.weight"
        
        # 将 unfused 的 gate_up_proj 张量切分
        # print(fused_state_dict[fused_k_gate].shape, v.shape)
        gate_v = v[:, :, :v.shape[-1] // 2].transpose(-1, -2)
        up_v = v[:, :, v.shape[-1] // 2:].transpose(-1, -2)
        
        # 更新我们准备好的字典
        # 使用 .contiguous() 是一个好习惯，确保内存布局正确
        fused_state_dict[fused_k_gate] = gate_v.contiguous()
        fused_state_dict[fused_k_up] = up_v.contiguous()

# 4. 将构建好的 state_dict 加载到 fused 模型中
# 这会**就地(in-place)**更新 model_fused 的权重
model_fused.load_state_dict(fused_state_dict)

# 5. 现在保存模型，它将包含正确的、转换后的权重
model_fused.save_pretrained("/opt/tiger/thu_qwenvl3/output/Qwen3-VL-30B-A3B-Instruct-Fused")

print("Model conversion and saving completed successfully!")