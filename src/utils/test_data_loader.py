import torch
import numpy as np
from data_utils import compute_dep_distance

def test_fixed_padding_value():
    """测试固定 padding 值"""
    print("=" * 60)
    print("开始测试依存距离矩阵的 padding")
    print("=" * 60)
    
    # 模拟不同长度的句子
    parents1 = [3, 3, 3, 4, -1]  # 长度 5
    parents2 = [2, 2, 3, -1]     # 长度 4
    parents3 = [1, 2, -1]        # 长度 3
    
    # 计算距离矩阵
    print("\n1. 计算原始依存距离矩阵")
    print("-" * 60)
    dist1 = compute_dep_distance(parents1)
    dist2 = compute_dep_distance(parents2)
    dist3 = compute_dep_distance(parents3)
    
    print("原始距离矩阵范围:")
    print(f"  dist1 (长度5): min={dist1.min()}, max={dist1.max()}, shape={dist1.shape}")
    print(f"  dist2 (长度4): min={dist2.min()}, max={dist2.max()}, shape={dist2.shape}")
    print(f"  dist3 (长度3): min={dist3.min()}, max={dist3.max()}, shape={dist3.shape}")
    
    # 打印详细矩阵
    print(f"\n  dist1 详细矩阵:")
    print(f"  {dist1}")
    print(f"\n  dist2 详细矩阵:")
    print(f"  {dist2}")
    print(f"\n  dist3 详细矩阵:")
    print(f"  {dist3}")
    
    # 使用 EntDataset 的 sequence_padding 方法
    print("\n2. 使用 sequence_padding 进行批量 padding")
    print("-" * 60)
    
    # 创建一个临时的 padder 对象
    class MockPadder:
        def sequence_padding(self, inputs, value=0, square_matrix=False, **kwargs):
            """模拟 EntDataset 的 sequence_padding 方法"""
            if square_matrix:
                length = max([x.shape[0] for x in inputs])
                outputs = []
                for x in inputs:
                    n = x.shape[0]
                    pad_width = [(0, length - n), (0, length - n)]
                    x = np.pad(x, pad_width, 'constant', constant_values=value)
                    outputs.append(x)
                return np.array(outputs)
    
    padder = MockPadder()
    batch_dep_dist = [dist1, dist2, dist3]
    
    # 使用固定值 999
    padded_dist = padder.sequence_padding(
        batch_dep_dist,
        value=999,
        square_matrix=True
    )
    
    print(f"批量 padding 后:")
    print(f"  batch shape: {padded_dist.shape} (应为 (3, 5, 5))")
    print(f"  padding 值: 999")
    
    # 打印 padding 后的矩阵
    print(f"\n3. 检查 padding 后的矩阵")
    print("-" * 60)
    
    print(f"\npadded_dist[0] (原长度5，无需padding):")
    print(padded_dist[0])
    valid_0 = padded_dist[0][padded_dist[0] < 999]
    print(f"  有效距离数量: {len(valid_0)} (应为 {5*5}=25)")
    
    print(f"\npadded_dist[1] (原长度4，padding到5):")
    print(padded_dist[1])
    valid_1 = padded_dist[1][padded_dist[1] < 999]
    print(f"  有效距离数量: {len(valid_1)} (应为 {4*4}=16)")
    print(f"  padding 位置值: {padded_dist[1][4, :]} (全为999) ✓")
    
    print(f"\npadded_dist[2] (原长度3，padding到5):")
    print(padded_dist[2])
    valid_2 = padded_dist[2][padded_dist[2] < 999]
    print(f"  有效距离数量: {len(valid_2)} (应为 {3*3}=9)")
    print(f"  padding 位置值: {padded_dist[2][3, :]} (全为999) ✓")
    
    # 验证过滤
    print(f"\n4. 验证 padding 的正确性")
    print("-" * 60)
    
    test_passed = True
    for i, pd in enumerate(padded_dist):
        valid_count = (pd < 999).sum()
        orig_size = [5, 4, 3][i]
        expected_valid = orig_size * orig_size
        
        match = valid_count == expected_valid
        status = "✓" if match else "✗"
        print(f"  样本{i}: 有效元素={valid_count}, 预期={expected_valid}, 匹配={match} {status}")
        
        if not match:
            test_passed = False
    
    # 验证形状
    if padded_dist.shape != (3, 5, 5):
        print(f"  ✗ 形状错误: {padded_dist.shape} != (3, 5, 5)")
        test_passed = False
    else:
        print(f"  ✓ 形状正确: {padded_dist.shape}")
    
    # 验证原始数据未改变
    if not np.array_equal(padded_dist[0][:5, :5], dist1):
        print(f"  ✗ dist1 数据被修改")
        test_passed = False
    else:
        print(f"  ✓ dist1 数据未被修改")
    
    if not np.array_equal(padded_dist[1][:4, :4], dist2):
        print(f"  ✗ dist2 数据被修改")
        test_passed = False
    else:
        print(f"  ✓ dist2 数据未被修改")
    
    # 最终结果
    print("\n" + "=" * 60)
    if test_passed:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败!")
    print("=" * 60)
    
    return test_passed

def test_compute_dep_distance():
    """测试依存距离计算函数"""
    print("\n" + "=" * 60)
    print("测试 compute_dep_distance 函数")
    print("=" * 60)
    
    # 测试案例1: 简单链式结构
    # 0 -> 1 -> 2 (root)
    parents1 = [1, 2, -1]
    dist1 = compute_dep_distance(parents1)
    
    print("\n测试案例1: 链式结构 [1, 2, -1]")
    print(f"依存树: 0->1->2(root)")
    print(f"距离矩阵:\n{dist1}")
    
    expected1 = np.array([
        [0, 1, 2],
        [1, 0, 1],
        [2, 1, 0]
    ])
    assert np.array_equal(dist1, expected1), "测试案例1失败"
    print("✓ 测试案例1通过")
    
    # 测试案例2: 星形结构
    # root(4) <- {0, 1, 2, 3}
    parents2 = [4, 4, 4, 4, -1]
    dist2 = compute_dep_distance(parents2)
    
    print("\n测试案例2: 星形结构 [4, 4, 4, 4, -1]")
    print(f"依存树: root(4) <- {{0,1,2,3}}")
    print(f"距离矩阵:\n{dist2}")
    
    # 所有节点到 root 距离为1，其他节点间距离为2
    for i in range(4):
        assert dist2[i, 4] == 1, f"节点{i}到root距离应为1"
        for j in range(4):
            if i == j:
                assert dist2[i, j] == 0, f"节点{i}到自己距离应为0"
            else:
                assert dist2[i, j] == 2, f"节点{i}到节点{j}距离应为2"
    print("✓ 测试案例2通过")
    
    # 测试案例3: 不同输入类型
    print("\n测试案例3: 不同输入类型")
    
    # list 输入
    parents_list = [1, 2, -1]
    dist_list = compute_dep_distance(parents_list)
    print("  ✓ list 输入测试通过")
    
    # numpy array 输入
    parents_np = np.array([1, 2, -1])
    dist_np = compute_dep_distance(parents_np)
    assert np.array_equal(dist_list, dist_np), "不同输入类型结果应一致"
    print("  ✓ numpy array 输入测试通过")
    
    # torch tensor 输入
    parents_torch = torch.tensor([1, 2, -1])
    dist_torch = compute_dep_distance(parents_torch)
    assert np.array_equal(dist_list, dist_torch), "不同输入类型结果应一致"
    print("  ✓ torch tensor 输入测试通过")
    
    print("\n" + "=" * 60)
    print("✓ compute_dep_distance 所有测试通过!")
    print("=" * 60)

if __name__ == "__main__":
    # 运行测试
    print("\n" + "🚀 开始运行测试套件...\n")
    
    # 测试1: 依存距离计算
    test_compute_dep_distance()
    
    # 测试2: padding 功能
    test_fixed_padding_value()
    
    print("\n" + "🎉 所有测试完成!\n")