import torch

def quaternion_geodesic_loss(q_true, q_pred, eps=1e-7):
    # 强制输入四元数单位化
    q_true = torch.nn.functional.normalize(q_true, p=2, dim=-1)
    q_pred = torch.nn.functional.normalize(q_pred, p=2, dim=-1)
    
    # 计算点积（考虑符号对称性）
    dot_product = torch.sum(q_true * q_pred, dim=-1)
    dot_product = torch.abs(dot_product)  # 处理q和-q等价
    
    # 数值稳定处理
    dot_product = torch.clamp(dot_product, min=0.0, max=1.0 - eps)
    
    # 计算测地线角度（弧度）
    theta = 2 * torch.arccos(dot_product)
    
    # 返回批次平均损失
    return theta.mean()

def quaternion_geodesic_loss_1(q_true, q_pred):
    q_true = torch.nn.functional.normalize(q_true, p=2, dim=-1)
    q_pred = torch.nn.functional.normalize(q_pred, p=2, dim=-1)
        
    dot_product = torch.abs(torch.sum(q_true * q_pred, dim=-1))  # 取绝对值解决q和-q等价
    theta = 2 * torch.arccos(torch.clamp(dot_product, min=0.0, max=1.0 - 1e-7))
    return theta

q_true = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
q_pred = torch.tensor([[0.0, 1.0, 0.0, 0.0]], requires_grad=True)
loss = quaternion_geodesic_loss_1(q_true, q_pred)
loss.backward()
print(q_pred.grad)  # 应有合理非零梯度

print("q_true:", q_true)
print("q_pred:", q_pred)
print("loss:", loss.item())
print("q_pred.grad:", q_pred.grad)  # 应接近零向量