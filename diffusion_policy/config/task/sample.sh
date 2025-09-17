# 设定目标目录
TARGET_20="/metaworld_drawer-open_expert_20percent.zarr"
TARGET_10="/metaworld_drawer-open_expert_10percent.zarr"
TARGET_4="/metaworld_drawer-open_expert_4percent.zarr"

# 创建目标目录结构
mkdir -p $TARGET_20/data/{action,full_state,img,state}
mkdir -p $TARGET_10/data/{action,full_state,img,state}
mkdir -p $TARGET_4/data/{action,full_state,img,state}

# 20%抽样 - 每5个文件抽取1个
for i in $(seq 0 1 100); do
  # 不同目录使用不同的文件名格式
  img_filename="${i}.0.0.0"
  other_filename="${i}.0"
  
  # 复制所有相关文件
  cp -v action/$other_filename $TARGET_20/data/action/ 2>/dev/null
  cp -v full_state/$other_filename $TARGET_20/data/full_state/ 2>/dev/null
  cp -v img/$img_filename $TARGET_20/data/img/ 2>/dev/null
  cp -v state/$other_filename $TARGET_20/data/state/ 2>/dev/null
done

# 10%抽样 - 每10个文件抽取1个
for i in $(seq 0 1 50); do
  img_filename="${i}.0.0.0"
  other_filename="${i}.0"
  
  cp -v action/$other_filename $TARGET_10/data/action/ 2>/dev/null
  cp -v full_state/$other_filename $TARGET_10/data/full_state/ 2>/dev/null
  cp -v img/$img_filename $TARGET_10/data/img/ 2>/dev/null
  cp -v state/$other_filename $TARGET_10/data/state/ 2>/dev/null
done

# 4%抽样 - 每25个文件抽取1个
for i in $(seq 0 1 20); do
  img_filename="${i}.0.0.0"
  other_filename="${i}.0"
  
  cp -v action/$other_filename $TARGET_4/data/action/ 2>/dev/null
  cp -v full_state/$other_filename $TARGET_4/data/full_state/ 2>/dev/null
  cp -v img/$img_filename $TARGET_4/data/img/ 2>/dev/null
  cp -v state/$other_filename $TARGET_4/data/state/ 2>/dev/null
done

# 统计文件数量
echo "20%抽样目录文件统计："
find $TARGET_20 -type f | wc -l

echo "10%抽样目录文件统计："
find $TARGET_10 -type f | wc -l

echo "4%抽样目录文件统计："
find $TARGET_4 -type f | wc -l