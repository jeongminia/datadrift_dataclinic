import sys
from typing import Optional
from pymilvus import connections, utility

def milvus_rm(target: Optional[str] = None, host: str = "localhost", port: str = "19530") -> None:

    # Milvus 서버 연결
    connections.connect("default", host=host, port=port)

    if target:
        if utility.has_collection(target):
            utility.drop_collection(target)
            print(f"✅ Deleted collection: {target}")
        else:
            print(f"❌ Collection '{target}' does not exist.")
    else:
        all_collections = utility.list_collections()
        if not all_collections:
            print("📭 삭제할 컬렉션이 없습니다.")
            return
        print("📦 삭제할 컬렉션 목록:")
        for name in all_collections:
            print(f" - {name}")
        for name in all_collections:
            utility.drop_collection(name)
            print(f"✅ Deleted collection: {name}")
        print("🎉 모든 컬렉션 삭제 완료!")

# CLI 실행 지원
if __name__ == "__main__":
    target_collection = sys.argv[1] if len(sys.argv) > 1 else None
    milvus_rm(target=target_collection)
