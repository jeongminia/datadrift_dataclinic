from pymilvus import connections, utility, Collection

# Milvus 서버 연결
connections.connect("default", host="localhost", port="19530")

# 컬렉션 목록 가져오기
collections = utility.list_collections()

if not collections:
    print("⚠️ 현재 Milvus에 존재하는 컬렉션이 없습니다.")
else:
    print(f"📦 현재 존재하는 {len(collections)}개 컬렉션 목록:\n")

    for name in collections:
        print(f"🔍 [컬렉션 이름] {name}")

        collection = Collection(name)
        collection.load()

        # 필드 정보 출력
        print("  📌 [스키마]")
        for field in collection.schema.fields:
            print(f"    - {field.name} ({field.dtype})")

        # 총 데이터 개수 출력
        print(f"  📊 [총 레코드 수] {collection.num_entities}")

        # set_type 값 고유 목록 확인
        try:
            results = collection.query(
                expr="", 
                output_fields=["set_type"], 
                limit=10000  # 필요에 따라 늘릴 수 있음
            )
            set_types = [res["set_type"] for res in results if "set_type" in res]
            unique_types = set(set_types)
            print(f"  🧷 [set_type 고유값] {unique_types}")
        except Exception as e:
            print(f"  ⚠️ set_type 조회 중 오류: {e}")
