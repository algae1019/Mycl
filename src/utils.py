'''
다양한 유틸리티 구현
'''


# 경로 자동화 기능
import os

# 프로젝트 루트 디렉터리 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#print("프로젝트 루트 경로:", PROJECT_ROOT)  # 루트 경로 확인

def get_path(*subdirs):
    """
    프로젝트 루트 디렉터리를 기준으로 하위 경로를 생성
    예: get_project_path("data", "train") -> /path/to/project/data/train
    """
    full_path = os.path.join(PROJECT_ROOT, *subdirs)
    return full_path