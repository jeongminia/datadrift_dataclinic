from pymilvus import connections, utility

# Milvus 서버에 연결
connections.connect("default", host="localhost", port="19530")

# 현재 존재하는 모든 컬렉션 이름 가져오기
all_collections = utility.list_collections()

print("📦 삭제할 컬렉션 목록:")
for name in all_collections:
    print(f" - {name}")

# 전체 컬렉션 삭제
for name in all_collections:
    utility.drop_collection(name)
    print(f"✅ Deleted collection: {name}")

print("🎉 모든 컬렉션 삭제 완료!")
