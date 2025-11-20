#!/bin/bash
# 单节点 CPU 版本 AlphaFold3 (xfold) 启动脚本
# 直接基于你给出的 120s baseline，外加“批量跑所有 JSON”能力。
#
# 用法：
#   sbatch run_af3_optimized.sh
#
# 注意：根据自己账号修改 INPUT_DIR / MODEL_DIR / OUTPUT_ROOT

#SBATCH -J af3_opt                 # 作业名称
#SBATCH -p cnmix                   # 使用队列
#SBATCH -N 1                       # 使用节点数
#SBATCH --ntasks=1                 # 单进程
#SBATCH --cpus-per-task=28         # 仅用单插槽 28 物理核，减小跨 NUMA 开销
#SBATCH -o logs/stdout.%j               # 标准输出
#SBATCH -e logs/stderr.%j               # 错误输出

########################################
# 1. 环境加载
########################################
source ~/.bashrc
conda activate af3                  # 替换为你的环境名

########################################
# 2. 线程与亲和性（单插槽 28 物理核）
########################################

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-28}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=CLOSE          # PyTorch Tuning Guide 推荐
export OMP_SCHEDULE=STATIC
export OMP_PLACES=cores

# Intel OpenMP 运行时推荐设置（若实际使用 libiomp）
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-1}"
export KMP_AFFINITY="${KMP_AFFINITY:-granularity=fine,compact,1,0}"

# 禁止其他 BLAS 再开线程，避免过订阅
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# oneDNN/MKL 指令集（6258R 支持 AVX-512）
export DNNL_MAX_CPU_ISA=AVX512_CORE

########################################
# 3. 可选：条件预加载 OpenMP / 分配器（存在才加载）
########################################

# 先切到 Intel OpenMP（libiomp），再接 jemalloc/TCMalloc
if [ -r /usr/lib/x86_64-linux-gnu/libiomp5.so ]; then
  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libiomp5.so${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
if [ -r /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 ]; then
  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4${LD_PRELOAD:+:${LD_PRELOAD}}"
elif [ -r /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then
  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

echo "[ENV] Start time: $(date '+%F %T')"
echo "[ENV] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[ENV] LD_PRELOAD=${LD_PRELOAD-}"

########################################
# 4. NUMA 策略（仅内存策略，避免与 Slurm cpuset 冲突）
########################################

# NUMA_MODE: auto | membind | none
NUMA_MODE="${NUMA_MODE:-auto}"
NUMA_NODE="${NUMA_NODE:-0}"
NUMA_PREFIX=()

if command -v numactl >/dev/null 2>&1; then
  case "${NUMA_MODE}" in
    membind)
      NUMA_PREFIX=(numactl --membind="${NUMA_NODE}")
      ;;
    auto|interleave)
      NUMA_PREFIX=(numactl --interleave=all)
      ;;
    none)
      NUMA_PREFIX=()
      ;;
  esac
fi

########################################
# 5. 路径配置：输入 / 模型 / 输出
########################################

# 解压 input.zip 后的 JSON 目录（请根据实际路径修改）
INPUT_DIR=/home/thuscc25team04/WORK/AlphaFold3/xfold/processed

# 模型参数目录（af3.bin 等）
MODEL_DIR=/WORK/sccomp/weights

# 输出根目录（每个测例一个子目录）
OUTPUT_ROOT=/home/thuscc25team04/WORK/AlphaFold3/xfold/output

mkdir -p "${OUTPUT_ROOT}"

########################################
# 6. 逐个测例运行（默认所有 JSON）
########################################

shopt -s nullglob
# JSON_FILES=("${INPUT_DIR}"/*.json)
JSON_FILES=("${INPUT_DIR}"/37aa_2JO9.json)
shopt -u nullglob

if [ "${#JSON_FILES[@]}" -eq 0 ]; then
  echo "[RUN] No JSON files found under ${INPUT_DIR}."
  exit 1
fi

for json in "${JSON_FILES[@]}"; do
  name=$(basename "${json}" .json)
  out_dir="${OUTPUT_ROOT}/${name}"

  echo "============================================"
  echo "[RUN] $(date '+%F %T') - Start ${name}"
  echo "[RUN] Input JSON: ${json}"
  echo "[RUN] Output dir: ${out_dir}"

  mkdir -p "${out_dir}"

  # 使用 --norun_data_pipeline，因为比赛已提供预处理 MSA & 模板
  # 使用 --nofastnn，只用 PyTorch CPU 实现
  "${NUMA_PREFIX[@]}" \
  python run_alphafold.py \
    --json_path="${json}" \
    --model_dir="${MODEL_DIR}" \
    --norun_data_pipeline \
    --output_dir="${out_dir}" \
    --nofastnn

  echo "[RUN] $(date '+%F %T') - Done ${name}"
done

echo "[ENV] End time: $(date '+%F %T')"
