def solve_n_queens(n):
    def is_safe(board, row, col):
        # 检查同一列
        for i in range(row):
            if board[i] == col:
                return False
            # 检查对角线
            if abs(board[i] - col) == abs(i - row):
                return False
        return True

    def backtrack(board, row):
        if row == n:
            solutions.append(board[:])
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                backtrack(board, row + 1)
                board[row] = -1  # 回溯

    solutions = []
    board = [-1] * n
    backtrack(board, 0)
    return solutions

def print_solutions(solutions):
    for sol in solutions:
        for row in sol:
            line = ['·'] * len(sol)
            line[row] = 'Q'
            print(' '.join(line))
        print()

if __name__ == "__main__":
    n = int(input("请输入棋盘大小(n): "))
    solutions = solve_n_queens(n)
    print(f"共有{len(solutions)}种解法")
    if solutions:
        print_solutions(solutions)
