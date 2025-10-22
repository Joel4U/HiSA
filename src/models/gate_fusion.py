import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UnifiedSpanFusion(nn.Module):
    """统一的Span融合模块"""
    def __init__(self, hidden_dim, fusion_type="adaptive", dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        if fusion_type == "simple":
            self.fusion = SimpleGateFusion(hidden_dim, dropout)
        elif fusion_type == "adaptive":
            self.fusion = AdaptiveWeightFusion(hidden_dim, dropout)
        elif fusion_type == "spatial":
            self.fusion = SpatialAwareFusion(hidden_dim, dropout)
        elif fusion_type == "multihead":
            self.fusion = MultiHeadFusion(hidden_dim, dropout)
        else:
            self.fusion = SimpleGateFusion(hidden_dim, dropout)
    
    def forward(self, u_scores, g_scores, scores):
        return self.fusion(u_scores, g_scores, scores)


class SimpleGateFusion(nn.Module):
    """简单高效的门控融合"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.gate_net = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, u_scores, g_scores, scores):
        combined = torch.cat([u_scores, g_scores], dim=1)
        gate = self.gate_net(combined)
        fused = gate * u_scores + (1 - gate) * g_scores
        return fused + self.alpha * scores


class AdaptiveWeightFusion(nn.Module):
    """自适应权重融合"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        
        # 学习每个输入的重要性
        self.importance_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 8, 1, 1)
        )
        
        # 门控融合
        self.gate_net = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.beta = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, u_scores, g_scores, scores):
        # 计算重要性权重
        w_u = self.importance_net(u_scores)
        w_g = self.importance_net(g_scores)
        w_s = self.importance_net(scores)
        
        # Softmax归一化
        weights = torch.cat([w_u, w_g, w_s], dim=1)
        weights = F.softmax(weights, dim=1)
        w_u, w_g, w_s = weights.chunk(3, dim=1)
        
        # 加权组合
        weighted = w_u * u_scores + w_g * g_scores
        
        # 门控融合
        combined = torch.cat([weighted, scores], dim=1)
        gate = self.gate_net(combined)
        
        fused = gate * weighted + (1 - gate) * scores
        
        # 残差连接
        return fused + self.beta * w_s * scores


class SpatialAwareFusion(nn.Module):
    """空间感知融合"""
    def __init__(self, hidden_dim, dropout=0.1, kernel_size=3):
        super().__init__()
        
        padding = kernel_size // 2
        
        # 局部特征提取
        self.local_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # 门控
        self.gate_net = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, u_scores, g_scores, scores):
        # 提取局部特征
        combined = torch.cat([u_scores, g_scores], dim=1)
        local_feat = self.local_conv(combined)
        
        # 门控融合
        gate_input = torch.cat([u_scores, g_scores, local_feat], dim=1)
        gate = self.gate_net(gate_input)
        
        # 融合
        fused = gate * u_scores + (1 - gate) * g_scores
        return fused + scores


class MultiHeadFusion(nn.Module):
    """多头融合"""
    def __init__(self, hidden_dim, dropout=0.1, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # 使用一个统一的门控网络，而不是多个
        self.gate_net = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 多头投影
        self.head_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, groups=num_heads)
        
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, u_scores, g_scores, scores):
        B, C, H, W = u_scores.shape
        
        # 计算门控
        combined = torch.cat([u_scores, g_scores], dim=1)
        gates = self.gate_net(combined)  # [B, C, H, W]
        
        # 多头处理
        # 将通道维度reshape为[num_heads, head_dim]
        gates_heads = gates.view(B, self.num_heads, self.head_dim, H, W)
        u_heads = u_scores.view(B, self.num_heads, self.head_dim, H, W)
        g_heads = g_scores.view(B, self.num_heads, self.head_dim, H, W)
        
        # 每个头独立融合
        fused_heads = gates_heads * u_heads + (1 - gates_heads) * g_heads
        
        # 合并头
        fused = fused_heads.view(B, C, H, W)
        
        # 应用头投影
        fused = self.head_proj(fused)
        
        # 残差连接
        output = fused + scores
        
        return self.dropout(output)


# 更简单的多头实现（推荐）
class SimpleMultiHeadFusion(nn.Module):
    """简化的多头融合 - 更稳定"""
    def __init__(self, hidden_dim, dropout=0.1, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 为每个头创建独立的门控，但参数共享
        self.gate_net = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 头间交互
        self.cross_head = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
    def forward(self, u_scores, g_scores, scores):
        # 标准门控融合
        combined = torch.cat([u_scores, g_scores], dim=1)
        gate = self.gate_net(combined)
        
        # 融合
        fused = gate * u_scores + (1 - gate) * g_scores
        
        # 头间交互
        fused = self.cross_head(fused)
        
        # 残差
        return fused + scores


# 性能测试
if __name__ == "__main__":
    import time
    
    # 测试参数
    bsz, dim, L = 16, 768, 50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 创建测试数据
    u_scores = torch.randn(bsz, dim, L, L, device=device)
    g_scores = torch.randn(bsz, dim, L, L, device=device)
    scores = torch.randn(bsz, dim, L, L, device=device)
    
    # 测试不同融合策略
    strategies = ["simple", "adaptive", "spatial", "multihead"]
    
    print("性能对比测试:")
    print("-" * 80)
    print(f"{'Strategy':<15} {'Parameters':<15} {'Time (ms)':<15} {'Memory (MB)':<15}")
    print("-" * 80)
    
    results = []
    
    for strategy in strategies:
        try:
            model = UnifiedSpanFusion(dim, fusion_type=strategy).to(device)
            model.eval()
            
            # 参数量
            params = sum(p.numel() for p in model.parameters())
            
            # 预热
            with torch.no_grad():
                _ = model(u_scores, g_scores, scores)
            
            # 速度测试
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.time()
            
            with torch.no_grad():
                for _ in range(50):
                    _ = model(u_scores, g_scores, scores)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            elapsed = (time.time() - start) / 50 * 1000
            
            # 内存使用
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = model(u_scores, g_scores, scores)
                memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            else:
                memory = 0
                
            print(f"{strategy:<15} {params:<15,} {elapsed:<15.2f} {memory:<15.2f}")
            
            results.append({
                'strategy': strategy,
                'params': params,
                'time': elapsed,
                'memory': memory
            })
            
        except Exception as e:
            print(f"{strategy:<15} Error: {str(e)}")
    
    # 推荐
    print("\n" + "="*80)
    print("推荐使用:")
    print("1. 一般情况: simple (最快速，参数少)")
    print("2. 复杂任务: adaptive (自适应权重)")
    print("3. 长文本: spatial (考虑局部特征)")
    print("4. 高性能需求: multihead (多视角融合)")
