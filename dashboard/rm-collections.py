# dashboard/rm-collections.py

import sys
from pymilvus import connections, utility

# Milvus 서버에 연결
connections.connect("default", host="localhost", port="19530")

# 인자로 컬렉션 이름 받기
if len(sys.argv) > 1:
    target = sys.argv[1]
    if utility.has_collection(target):
        utility.drop_collection(target)
        print(f"✅ Deleted collection: {target}")
    else:
        print(f"❌ Collection '{target}' does not exist.")
else:
    # 인자가 없으면 전체 삭제
    all_collections = utility.list_collections()
    print("📦 삭제할 컬렉션 목록:")
    for name in all_collections:
        print(f" - {name}")
    for name in all_collections:
        utility.drop_collection(name)
        print(f"✅ Deleted collection: {name}")
    print("🎉 모든 컬렉션 삭제 완료!")
