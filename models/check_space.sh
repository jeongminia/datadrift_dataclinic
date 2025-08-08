#!/bin/bash
REQUIRED_SPACE_GB=50

echo "📦 Disk Space Information:"
df -h --output=source,size,used,avail,pcent,target | tail -n +2

echo
# 루트(/) 파티션 기준으로만 설치 가능 여부 판단
AVAILABLE_SPACE=$(df / | tail -1 | awk '{print $4}')
AVAILABLE_SPACE_GB=$((AVAILABLE_SPACE / 1024 / 1024))

echo " "
if [ $AVAILABLE_SPACE_GB -lt $REQUIRED_SPACE_GB ]; then
    echo "⚠️  경고: 사용 가능한 공간이 부족합니다."
    echo "   필요: ${REQUIRED_SPACE_GB}GB, 사용가능: ${AVAILABLE_SPACE_GB}GB"
    exit 1
else
    echo "✅ 충분한 저장공간이 있습니다."
fi
