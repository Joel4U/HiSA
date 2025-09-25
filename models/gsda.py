import math, torch
from torch import nn as nn
import torch.nn.functional as F

from timm.layers import create_act_layer, get_act_layer
from timm.layers import create_conv2d
from timm.layers import make_divisible
from timm.layers.mlp import ConvMlp
from timm.layers.norm import LayerNorm2d

device = torch.device("cuda:2")

class GatherExcite(nn.Module):
    def __init__(
            self, channels, feat_size=None, extra_params=False, extent=0, use_mlp=True,
            rd_ratio=1./16, rd_channels=None,  rd_divisor=1, add_maxpool=False,
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, gate_layer='sigmoid'):
        super(GatherExcite, self).__init__()
        self.add_maxpool = add_maxpool
        act_layer = get_act_layer(act_layer)
        self.extent = extent
        if extra_params:
            self.gather = nn.Sequential()
            if extent == 0:
                assert feat_size is not None, 'spatial feature size must be specified for global extent w/ params'
                self.gather.add_module(
                    'conv1', create_conv2d(channels, channels, kernel_size=feat_size, stride=1, depthwise=True))
                if norm_layer:
                    self.gather.add_module(f'norm1', nn.BatchNorm2d(channels))
            else:
                assert extent % 2 == 0
                num_conv = int(math.log2(extent))
                for i in range(num_conv):
                    self.gather.add_module(
                        f'conv{i + 1}',
                        create_conv2d(channels, channels, kernel_size=3, stride=2, depthwise=True))
                    if norm_layer:
                        self.gather.add_module(f'norm{i + 1}', nn.BatchNorm2d(channels))
                    if i != num_conv - 1:
                        self.gather.add_module(f'act{i + 1}', act_layer(inplace=True))
        else:
            self.gather = None
            if self.extent == 0:
                self.gk = 0
                self.gs = 0
            else:
                assert extent % 2 == 0
                self.gk = self.extent * 2 - 1
                self.gs = self.extent

        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.mlp = ConvMlp(channels, rd_channels, act_layer=act_layer) if use_mlp else nn.Identity()
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        size = x.shape[-2:]
        if self.gather is not None:
            x_ge = self.gather(x)
        else:
            if self.extent == 0:
                # global extent
                x_ge = x.mean(dim=(2, 3), keepdims=True)
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * x.amax((2, 3), keepdim=True)
            else:
                x_ge = F.avg_pool2d(
                    x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2, count_include_pad=False)
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * F.max_pool2d(x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2)
        x_ge = self.mlp(x_ge) 
        if x_ge.shape[-1] != 1 or x_ge.shape[-2] != 1:
            x_ge = F.interpolate(x_ge, size=size)
        return x * self.gate(x_ge)

class LayerNorm(nn.Module):
    def __init__(self, shape=(1, 7, 1, 1), dim_index=1):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.dim_index = dim_index
        self.eps = 1e-6

    def forward(self, x):
        """
        :param x: bsz x dim x max_len x max_len
        :param mask: bsz x dim x max_len x max_len, 为1的地方为pad
        :return:
        """
        u = x.mean(dim=self.dim_index, keepdim=True)
        s = (x - u).pow(2).mean(dim=self.dim_index, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x

class SelfAttentionBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels=None, bn_type=None):
        super(SelfAttentionBlock2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if out_channels == None:
            self.out_channels = in_channels

        # self.key_channels = 1
        self.key_channels = in_channels // 8
        self.value_channels = in_channels

        self.f_key = nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, bias=False)
        self.f_query = nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, bias=False)

        self.f_value = nn.Conv2d(self.in_channels, self.value_channels, kernel_size=1, bias=False)

        self.W = nn.Sequential(
            nn.Conv2d(self.value_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )
        # v1:
        # self.bias_conv_1 = nn.Conv2d(in_channels, self.key_channels, kernel_size=1, bias=False)
        # self.bias_conv_2 = nn.Conv2d(in_channels, self.key_channels, kernel_size=1, bias=False)
        # v2:
        self.bias_conv_1 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Conv2d(in_channels, self.key_channels, kernel_size=1, bias=False),
            nn.ReLU()
        )
        self.bias_conv_2 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Conv2d(in_channels, self.key_channels, kernel_size=1, bias=False),
            nn.ReLU()
        )

        self.get_bias = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, padding=3 // 2, stride=1),
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.get_bias_sigmoid = nn.Sigmoid()
        # self.ge = GatherExcite(in_channels) 
        self.ge = GatherExcite(in_channels)     
        # self.bias_gamma = nn.Parameter(torch.zeros(1)) 
        self.gamma = nn.Parameter(torch.zeros(1)) 

    def forward(self, x, dep_mask=None):
        """
        x: [batch_size, channels, h, w]
        dep_mask: [batch_size, h*w, h*w]
        """
        # batch_size, h, w = x.size(0), x.size(2), x.size(3)
        batch_size, channels, h, w = x.size()
        # 生成Q, K, V        
        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)  # [batch, h*w, key_channels]

        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1) # [batch, h*w, key_channels]

        key = self.ge(x)
        key = self.f_key(key).view(batch_size, self.key_channels, -1)
        # w/o GE
        # key = self.f_key(x).view(batch_size, self.key_channels, -1)      

        sim_map = torch.matmul(query, key)
        # 如果提供了依存关系mask，应用它
        if dep_mask is not None:
            # 将无关位置设为极小值
            sim_map = sim_map.masked_fill(dep_mask == 0, float('-inf'))
        # 计算自适应bias
        bias_1 = self.bias_conv_1(x).view(batch_size, self.key_channels, -1).permute(0, 2, 1)
        bias_2 = self.bias_conv_2(x).view(batch_size, self.key_channels, -1)
        bias = torch.matmul(bias_1, bias_2)     # [batch, h*w, h*w]
        bias = self.get_bias(bias.unsqueeze(1)).squeeze(1)
        # 应用softmax
        if dep_mask is not None: 
            # 确保每行至少有一个有效值
            row_sum = dep_mask.sum(dim=-1, keepdim=True)
            sim_map = torch.where(
                row_sum > 0,
                F.softmax(sim_map, dim=-1),
                torch.zeros_like(sim_map)
            )
            sim_map = sim_map * dep_mask.float()       # 再次应用mask确保无效位置为0
        else:
            sim_map = F.softmax(sim_map * bias, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, h, w)
        context = self.W(context)
        return self.gamma * context + x

class GlobalSDAware(nn.Module):
    def __init__(self, channel, use_dep_mask=True):
        super(GlobalSDAware, self).__init__()
        self.use_dep_mask = use_dep_mask
        # 不使用Sequential，因为需要传递额外参数
        self.attention_block_1 = SelfAttentionBlock2D(channel)
        self.layer_norm_1 = LayerNorm((1, channel, 1, 1), dim_index=1)
        self.activation_1 = nn.GELU()
        self.attention_block_2 = SelfAttentionBlock2D(channel)
        self.layer_norm_2 = LayerNorm((1, channel, 1, 1), dim_index=1)
        self.activation_2 = nn.GELU()

    def forward(self, x, head_dep_mat=None):
        """
        x: [batch_size, hidden_dim, n, n] span representations
        head_dep_mat: [batch_size, n, n] head indices for each span
        """
        dependency_mask = None
        # 生成依存关系遮蔽矩阵
        if self.use_dep_mask and head_dep_mat is not None:
            dependency_mask = self.generate_dependency_mask(head_dep_mat)
        # 第一个注意力块
        out1 = self.attention_block_1(x, dependency_mask)
        out1 = self.layer_norm_1(out1)
        out1 = self.activation_1(out1)
        # 第二个注意力块
        out2 = self.attention_block_2(out1, dependency_mask)
        out2 = self.layer_norm_2(out2)
        out2 = self.activation_2(out2)
        # out1 = self.conv_block_1(x)
        # out1 = self.conv_block_2(out1)
        return out2

    def generate_dependency_mask(self, head_matrix):
        """
        生成span间的依存关系mask
        head_matrix: [batch_size, n, n] 
        return: [batch_size, n*n, n*n] flattened mask
        """
        batch_size, n, _ = head_matrix.shape
        device = head_matrix.device
        
        # Flatten head matrix to [batch_size, n*n]
        head_flat = head_matrix.view(batch_size, -1)
        
        # 创建有效span的mask
        valid_mask = (head_flat != -1).float()  # [batch_size, n*n]
        
        # 初始化mask为对角矩阵（自注意力）
        mask = torch.eye(n * n, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 为了检查span包含关系，需要创建span起止位置矩阵
        positions = torch.arange(n, device=device)
        start_pos = positions.unsqueeze(1).expand(-1, n).reshape(-1)  # [n*n]
        end_pos = positions.unsqueeze(0).expand(n, -1).reshape(-1)    # [n*n]
        
        # 对每个batch处理
        for b in range(batch_size):
            heads = head_flat[b]  # [n*n]
            valid = valid_mask[b]  # [n*n]
            
            # 扩展为矩阵形式
            heads_i = heads.unsqueeze(1)  # [n*n, 1]
            heads_j = heads.unsqueeze(0)  # [1, n*n]
            
            valid_i = valid.unsqueeze(1)  # [n*n, 1]
            valid_j = valid.unsqueeze(0)  # [1, n*n]
            
            # 两个span都有效
            both_valid = valid_i * valid_j  # [n*n, n*n]
            
            # 条件1&2: span的head在另一个span内
            # span j 包含 span i 的 head
            start_j = start_pos.unsqueeze(0)  # [1, n*n]
            end_j = end_pos.unsqueeze(0)      # [1, n*n]
            contains_i_head = (start_j <= heads_i) & (heads_i <= end_j)
            
            # span i 包含 span j 的 head
            start_i = start_pos.unsqueeze(1)  # [n*n, 1]
            end_i = end_pos.unsqueeze(1)      # [n*n, 1]
            contains_j_head = (start_i <= heads_j) & (heads_j <= end_i)
            
            # 条件3: 相同的head
            same_head = (heads_i == heads_j)
            
            # 合并所有条件
            dependency = (contains_i_head | contains_j_head | same_head).float()
            mask[b] = mask[b] + dependency * both_valid
        
        # 确保mask是二值的，并转换到正确的设备
        mask = (mask > 0).float().to(device)
        return mask

    
if __name__ == '__main__':
    # print("=== 测试GatherExcite模块 ===")
    # input = torch.randn(50, 512, 7, 7).to(device)
    # GE = GatherExcite(512).to(device)
    # output = GE(input)
    # print(f"GatherExcite - 输入形状: {input.shape}, 输出形状: {output.shape}")
    
    print("\n=== 测试GlobalSDAware模块 ===")
    n = 9
    # tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"], 句子长度 n=9
    # -1表示根，其他数字表示头词在句子中的索引
    # dep_heads = [3, 3, 3, 4, -1, 8, 8, 8, 4] # jumps是根，quick修饰fox等
    # 直接给出head矩阵（上三角部分，-1表示无效）
    # head_matrix[i,j] 表示 span(i,j+1) 的head词索引
    head_matrix = torch.tensor([
        [ 0,  1,  1,  3,  4,  4,  4,  4,  4],  # row 0: spans starting at "The"
        [-1,  1,  1,  3,  4,  4,  4,  4,  4],  # row 1: spans starting at "quick"
        [-1, -1,  2,  3,  4,  4,  4,  4,  4],  # row 2: spans starting at "brown"
        [-1, -1, -1,  3,  4,  4,  4,  4,  4],  # row 3: spans starting at "fox"
        [-1, -1, -1, -1,  4,  4,  4,  4,  4],  # row 4: spans starting at "jumps"
        [-1, -1, -1, -1, -1,  5,  5,  7,  8],  # row 5: spans starting at "over"
        [-1, -1, -1, -1, -1, -1,  6,  7,  8],  # row 6: spans starting at "the"
        [-1, -1, -1, -1, -1, -1, -1,  7,  8],  # row 7: spans starting at "lazy"
        [-1, -1, -1, -1, -1, -1, -1, -1,  8],  # row 8: spans starting at "dog"
    ]).to(device)
    # 创建span表示 [batch_size, hidden_dim, n, n]
    span_features=torch.randn(2,512,n,n).to(device) # batch_size:2, hidden_dim:512, n:9   
    # 创建head矩阵 [batch_size, n, n]
    head_matrix = head_matrix.unsqueeze(0).repeat(2, 1, 1).to(device)
    model = GlobalSDAware(channel=512, use_dep_mask=True).to(device)
    output = model(span_features, head_dep_mat=head_matrix)
    print(f"输入形状: {span_features.shape}")
    print(f"Head矩阵形状: {head_matrix.shape}")
    print(f"使用依存mask的输出形状: {output.shape}")
    
    # 测试不使用mask的情况
    output_no_mask = model(span_features, head_dep_mat=None)
    print(f"不使用mask的输出形状: {output_no_mask.shape}")

    # 3. 验证dependency mask生成
    print("\n=== 验证Dependency Mask ===")
    dep_mask = model.generate_dependency_mask(head_matrix)
    print(f"Dependency mask形状: {dep_mask.shape}")

    # 显示一个小例子：查看前几个span之间的依存关系
    print("\n查看span(0,1)与其他span的依存关系:")
    idx_01 = 0 * n + 0  # span "The"
    batch_idx = 0  # 查看第一个batch
    # 定义span名称的辅助函数
    def get_span_name(i, j):
        words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        if i <= j < len(words):
            return " ".join(words[i:j+1])
        return f"span({i},{j+1})"
    print(f"Span(0,1) = '{get_span_name(0, 0)}'")

   # 检查与其他span的依存关系
    for flat_idx in range(min(10, n*n)):  # 查看前10个span
        if dep_mask[batch_idx, idx_01, flat_idx] > 0:
            i = flat_idx // n
            j_span = flat_idx % n
            # 只显示有效的span
            if i <= j_span and head_matrix[batch_idx, i, j_span] != -1:
                span_head = head_matrix[batch_idx, i, j_span].item()
                print(f"  - 与span({i},{j_span+1}) '{get_span_name(i, j_span)}' 有依存关系, head={span_head}")
    
    # 额外验证：查看一个更复杂的span
    print(f"\n查看span(1,3)与其他span的依存关系:")
    idx_13 = 1 * n + 2  # span "quick brown fox"
    print(f"Span(1,3) = '{get_span_name(1, 2)}'")
    
    for flat_idx in range(min(15, n*n)):
        if dep_mask[batch_idx, idx_13, flat_idx] > 0:
            i = flat_idx // n
            j_span = flat_idx % n
            if i <= j_span and head_matrix[batch_idx, i, j_span] != -1:
                span_head = head_matrix[batch_idx, i, j_span].item()
                print(f"  - 与span({i},{j_span+1}) '{get_span_name(i, j_span)}' 有依存关系, head={span_head}")
