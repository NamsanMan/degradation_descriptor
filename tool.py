import os
import shutil
import math

def copy_odd_files(source_dir, dest_dir):
    """
    소스 디렉토리에서 파일 목록을 사전순으로 정렬한 뒤,
    홀수 번째 파일들만 대상 디렉토리로 복사합니다.
    대상 디렉토리가 없으면 생성합니다.
    """
    
    # 1. 대상 디렉토리 존재 확인 및 생성
    if not os.path.exists(dest_dir):
        try:
            os.makedirs(dest_dir)
            print(f"생성된 디렉토리: {dest_dir}")
        except OSError as e:
            print(f"오류: 디렉토리 생성 실패. {e}")
            return

    # 2. 소스 디렉토리에서 파일 목록 가져오기
    try:
        all_items = os.listdir(source_dir)
        # 디렉토리를 제외한 실제 파일만 필터링
        files = [f for f in all_items if os.path.isfile(os.path.join(source_dir, f))]
    except FileNotFoundError:
        print(f"오류: 소스 디렉토리를 찾을 수 없습니다. {source_dir}")
        return
    except NotADirectoryError:
        print(f"오류: 소스 경로가 디렉토리가 아닙니다. {source_dir}")
        return
    except PermissionError:
        print(f"오류: 소스 디렉토리에 접근 권한이 없습니다. {source_dir}")
        return

    # 3. 파일 목록을 사전순으로 정렬
    files.sort()

    # 4. 홀수 번째 파일 복사 (1번째, 3번째, 5번째...)
    # 리스트 인덱스는 0부터 시작하므로, 0, 2, 4... 인덱스가 홀수 번째 파일임.
    copied_count = 0
    for i, filename in enumerate(files):
        if i % 2 == 0:  # 인덱스가 짝수(0, 2, 4...)인 경우가 홀수 번째 파일
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            
            try:
                # copy2는 메타데이터(수정 시간 등)도 함께 복사하며, 덮어쓰기를 지원
                shutil.copy2(source_path, dest_path)
                print(f"복사 완료 ({i+1}번째): {filename}")
                copied_count += 1
            except Exception as e:
                print(f"오류: {filename} 복사 실패. {e}")

    print(f"\n총 {len(files)}개의 파일 중 {copied_count}개의 홀수 번째 파일 복사를 완료했습니다.")

# --- 실행 ---
# 사용 예시:
# source_folder = r"C:\path\to\source_directory"
# destination_folder = r"C:\path\to\destination_directory"

# # Windows 경로 사용 시 'r' 접두사(raw string)를 붙이거나 '\'를 '\\'로 이스케이프 처리해야 합니다.
source_folder = r"D:\LAB\datasets\project_use\CamVid_12_2Fold_v4\B_set\train\images" 
destination_folder = r"D:\LAB\datasets\project_use\CamVid_12_2Fold_HR_LR_mix\B_set\train\images"

# 테스트용 더미 파일 생성 (필요시 주석 해제)
# if not os.path.exists(source_folder):
#     os.makedirs(source_folder)
#     for i in range(1, 11):
#         with open(os.path.join(source_folder, f"file_{i:02d}.txt"), 'w') as f:
#             f.write(f"this is file {i}")

copy_odd_files(source_folder, destination_folder)