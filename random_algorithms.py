import random
import math

def majority_element(arr, k=20):
    """
    使用随机算法找出数组中的主元素(出现次数超过一半的元素)
    参数:
        arr: 输入数组
        k: 随机采样次数，默认20次
    返回:
        主元素(如果存在)，否则None
    """
    n = len(arr)
    if n == 0:
        return None
    
    for _ in range(k):
        candidate = random.choice(arr)
        count = 0
        for num in arr:
            if num == candidate:
                count += 1
                if count > n // 2:
                    return candidate
    return None

def is_prime(n, k=5):
    """
    使用Miller-Rabin算法测试一个数是否为素数
    参数:
        n: 要测试的数
        k: 测试轮数，默认5次
    返回:
        True如果可能是素数，False如果是合数
    """
    if n <= 1:
        return False
    elif n <= 3:
        return True
    elif n % 2 == 0:
        return False
    
    # 将n-1表示为d*2^s
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        
        for __ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

# 测试用例
if __name__ == "__main__":
    # 测试主元素算法
    arr1 = [2, 2, 3, 2, 4, 2, 5, 2, 2]
    print("主元素测试:", majority_element(arr1))  # 应该输出2
    
    # 测试素数算法
    print("素数测试:")
    print(7, "->", is_prime(7))    # True
    print(15, "->", is_prime(15))  # False
    print(101, "->", is_prime(101)) # True
