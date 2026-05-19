def max_programs_on_tape(program_lengths, tape_length):
    """
    使用贪心算法计算磁带上最多可以存储的程序数量
    
    参数:
        program_lengths (list): 每个程序的长度列表
        tape_length (int): 磁带的总长度
        
    返回:
        int: 最多可以存储的程序数量
    """
    # 按程序长度升序排序
    sorted_lengths = sorted(program_lengths)
    total = 0
    count = 0
    
    for length in sorted_lengths:
        if total + length <= tape_length:
            total += length
            count += 1
        else:
            break
    return count

if __name__ == "__main__":
    # 示例用法
    programs = [5, 3, 8, 2, 7, 1, 4]
    L = 10
    print(f"最多可以存储的程序数量: {max_programs_on_tape(programs, L)}")
