def print_bitboard(name: str, decimal_str: str, bit_length: int = 128, row_size: int = 9):
    """
    거대한 10진수 문자열을 2진수로 변환하고 보드 형태(2D)로 출력합니다.
    """
    # 1. 문자열을 파이썬 정수로 변환 후 2진수 문자열로 변경 ('0b' 제거)
    binary_str = bin(int(decimal_str))[2:]
    
    # 2. 지정된 비트 길이(예: 128비트)에 맞게 앞에 0을 채움
    padded_binary = binary_str.zfill(bit_length)
    
    # 3. 비트보드는 보통 오른쪽(LSB)이 0번 인덱스이므로, 
    # 보기 편하게 인덱스 0번이 왼쪽 위로 오도록 문자열을 뒤집음 (엔진 구현에 따라 다를 수 있음)
    reversed_binary = padded_binary[::-1]
    
    print(f"=== {name} ===")
    print(f"Raw Decimal: {decimal_str}")
    print(f"Raw Binary : {padded_binary}\n")
    print(f"Board View (Row size: {row_size}):")
    
    # 4. 행 크기에 맞춰 잘라서 출력 (1은 ■, 0은 □ 로 표시하면 보기 편합니다)
    for i in range(0, bit_length, row_size):
        chunk = reversed_binary[i : i + row_size]
        if not chunk:
            break
        
        # 0과 1을 시각적으로 뚜렷하게 변환
        visual_chunk = chunk.replace('0', '· ').replace('1', '■ ')
        print(f"Row {i//row_size:2d}: {visual_chunk}")
    print("\n")


# 테스트 데이터
walls_h_str = "1266412660188944021221804082175"
walls_v_str = "635684025334390696938229466625"

# 가로 벽(walls_h)이 총 몇 비트(bit_length)로 관리되는지, 
# 한 줄(row_size)이 몇 칸인지 C++ 엔진 설정에 맞게 조절해 주세요.
# (예시: 128비트, 한 줄에 9칸으로 가정)
print_bitboard("walls_h", walls_h_str, bit_length=128, row_size=10)
print_bitboard("walls_v", walls_v_str, bit_length=128, row_size=10)