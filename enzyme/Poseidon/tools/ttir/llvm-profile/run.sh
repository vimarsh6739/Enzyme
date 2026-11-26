#!/usr/bin/env bash
# Takes the output directory from prepare.py. Use the same GPU throughout.
set -eu -o pipefail
: "${LLVM_BIN:?LLVM 21 bin directory}" "${POSEIDON_RELEASE:?unpacked Poseidon LLVM 21 release}" "${NVCC:?CUDA 12.9 nvcc}"
src=$(cd "$(dirname "$0")" && pwd)
poseidon=$(cd "$src/../../.." && pwd)
cd "${1:?directory containing baseline.ll and marked.ll}"
model="$poseidon/cost_models/cm_sm_120_RTX5090"
plugin="$POSEIDON_RELEASE/lib/Poseidon-21.so"
compile=("$LLVM_BIN/clang++" -cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_120 -O1 -x ir
         "-fpass-plugin=$plugin" -load "$plugin" -emit-llvm)
cuda=("$NVCC" -arch=sm_120)
mkdir -p profiles

# NVCC registers the CUDA sample kernel. Link the emitted Triton entry as its
# device callee so the application and the profile writer share the fatbinary.
device_object() {
    sed 's/define ptx_kernel void @entry/define void @entry/' "$1.ll" > "$1-callee.ll"
    "$LLVM_BIN/llc" -march=nvptx64 -mcpu=sm_120 -fp-contract=off "$1-callee.ll" -o "$1.ptx"
    "${cuda[@]}" -dc "$1.ptx" -o "$1.o"
}

"${compile[@]}" "-fpass-plugin=$POSEIDON_RELEASE/lib/ClangEnzyme-21.so" \
    -mllvm -poseidon-profile-generate marked.ll -o instrumented.ll 2> profile-build.log
grep -q 'call void @poseidonLogGradCUDA' instrumented.ll
device_object instrumented
"${cuda[@]}" -std=c++17 -dc "$poseidon/runtime/fpprofiler/FPProfilerCUDA.cu" \
    -I"$poseidon/../include" -o profiler.o
"${cuda[@]}" -DPROFILE -dc "$src/sample.cu" -o sample-profile.o
"${cuda[@]}" sample-profile.o instrumented.o profiler.o -o profile
POSEIDON_PROFILE_DIR="$PWD/profiles" ./profile
"${cuda[@]}" -dc "$src/sample.cu" -o sample.o

for variant in baseline 1e-17 1e-12 1e-3; do
    if [[ "$variant" != baseline ]]; then
        mkdir -p "explain-$variant"
        "${compile[@]}" -mllvm "-poseidon-profile-use=$PWD/profiles" \
            -mllvm "-poseidon-cost-model=$model.csv" -mllvm -poseidon-enable-pt \
            -mllvm -poseidon-enable-herbie -mllvm "-poseidon-herbie-binary=$POSEIDON_RELEASE/herbie/bin/herbie" \
            -mllvm "-poseidon-herbie-platform=$model.herbie.rkt" -mllvm -poseidon-herbie-num-threads=2 \
            -mllvm -poseidon-herbie-num-pts=256 -mllvm -poseidon-herbie-num-iters=4 \
            -mllvm -poseidon-herbie-num-enodes=4000 -mllvm -poseidon-herbie-timeout=45 \
            -mllvm -poseidon-herbie-subgraph-timeout=90 -mllvm "-poseidon-cache=$PWD/search-cache" \
            -mllvm "-poseidon-explain=$PWD/explain-$variant" -mllvm "-poseidon-tau=$variant" \
            -mllvm -poseidon-tau-cheapest -mllvm -poseidon-print marked.ll -o "$variant.ll" 2>"solve-$variant.log"
    fi
    device_object "$variant"
    "${cuda[@]}" sample.o "$variant.o" -o "eval-$variant"
    "./eval-$variant" > "$variant.csv"
done
