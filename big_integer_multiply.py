def karatsuba(x, y):
    """Karatsuba算法实现大整数乘法"""
    # 基础情况：当数字足够小时直接相乘
    if x < 10 or y < 10:
        return x * y
    
    # 计算数字长度
    n = max(len(str(x)), len(str(y)))
    m = n // 2
    
    # 分割数字
    high1, low1 = divmod(x, 10**m)
    high2, low2 = divmod(y, 10**m)
    
    # 递归计算三个部分
    z0 = karatsuba(low1, low2)
    z1 = karatsuba((low1 + high1), (low2 + high2))
    z2 = karatsuba(high1, high2)
    
    # 合并结果
    return (z2 * 10**(2*m)) + ((z1 - z2 - z0) * 10**m) + z0

# 测试用例
if __name__ == "__main__":
    # 测试两个大数相乘
    num1 = 12345678901234567890
    num2 = 98765432109876543210
    
    print(f"计算 {num1} × {num2}:")
    result = karatsuba(num1, num2)
    print(f"结果: {result}")
    print(f"验证: {num1 * num2 == result}")
