# LAM Blackwell 依赖兼容

上游 `aigc3d/LAM` 固定在 `339573649dd93df4cba8093a964e85a80d1b61f3`。只改 fork `zhting/LAM`。追踪器仍走 FaceBoxes 编译出的 `cpu_nms`。没有假检测器，也没有用 CPU 构建代替 GPU 通过。

## 版本：测过的和只在日志里见过的

| | 第一次 G4 诊断 | 第二次 G4 验证（`d4f1bfc`） | 这次离线修复 |
| --- | --- | --- | --- |
| Python | 3.13.16 | 3.13.16 | **3.13.18**（同一 3.13 ABI，补丁版本不同） |
| torch | **2.11.0+cu130**，插件 import 失败已测 | **2.14.1+cu130**，系统 torch，没有降级 | 本机没有 torch，没有装 |
| 结论 | matmul、GS、simple-knn 的 kernel PASS | matmul PASS；FaceBoxes CPU NMS PASS；nvdiffrast FAIL；PyTorch3D 3.893s 编译失败 | 复现并修了 ninja 发现和 C++ 标准。CUDA 编译与 kernel **UNTESTED** |

GS / simple-knn 的 PASS 属于 torch 2.11。torch 换成 2.14.1 之后没有再编，不能沿用。

## 第二次 G4 的两个真失败

1. **ninja**。venv 里已经装了 ninja 1.13.2。torch 2.14 的 `is_ninja_available()` 只执行 PATH 上的 `ninja --version`。引导脚本用绝对路径启动 `<venv>/bin/python`，没有 activate，于是 PATH 里没有 `<venv>/bin`。nvdiffrast 在建立 CUDA context 时抛 `Ninja is required`。PyTorch3D 则退回 distutils，日志写明找不到 ninja。
2. **C++ 标准**。退回 distutils 后的 nvcc 命令带有 `-std=c++17`（`setup.py` 对 host 和 nvcc 都写了 c++17）。torch 2.14 的 ATen 直接 `#error C++20 or later`。3.893 秒是编译错误，不是早先的 160 秒超时。torch 2.14 只有在参数里还没有 `-std=` 时才会自己补 c++20，所以不会盖掉这份写死的 c++17。ninja 找到之后，写进 `build.ninja` 的仍是同一对参数，host 和 nvcc 都会中招。

## 修复

- `lam_runtime.py` 装进 venv 的 `.pth`。绝对路径启动解释器时就会按 `sys.executable` 所在目录、`sysconfig` scripts、`ninja.BIN_DIR` 找到 ninja，跑一次 `--version`，再把该目录放进本进程的 `PATH`。子进程（ninja、以及 Python 直接拉起的编译器）继承这份环境。找不到可执行文件就抛错。
- nvdiffrast `22718580` 补丁仍保留 `load()` 的返回值、背面剔除（`RenderModeFlag_EnableBackfaceCulling` 与 `glEnable(GL_CULL_FACE)`）和不清空 `TORCH_CUDA_ARCH_LIST`。编译前同样要求能跑到 ninja。
- PyTorch3D `e4ad6e5` 的补丁把 host 的 `-std=c++17` 和 nvcc 的 `-std=c++17` 都改成 c++20，并保留 CUDA 13 的 visibility 开关。构建脚本在 checkout 之后重新打补丁；若 `setup.py` 里还剩 c++17 就失败退出。直接由 Python 拉起的编译命令还会把 c++14/17 收成单一的 c++20，避免 nvcc 收到两个 `-std`。ninja 自己 exec 绝对路径 nvcc 时靠的是 setup.py 里的参数，不是 PATH 包装。

## 这次离线实测

- 不 activate、PATH 只有 `/usr/bin:/bin`、用绝对路径启动 venv Python：`ninja` 是 `FileNotFoundError`。加上 runtime hook 之后版本是 `1.13.2`，子进程 PATH 含 `<venv>/bin`。`.pth` 在没有 `PYTHONPATH` 时同样生效。
- 用 G4 那条 nvcc 命令做参数改写：`-std=c++17` 变成 `-std=c++20`，CUDA 13 的 visibility 开关还在，只留一个 `-std`。本机没有 nvcc，**没有执行 nvcc**。
- 本机 `c++ -std=c++17` 经改写后真的编过一个要求 `__cplusplus >= 202002L` 的对象文件，**PASS**。
- FaceBoxes 合成框 CPU NMS 仍是 **PASS**，`[0, 2]`。
- PyTorch3D 与 nvdiffrast CUDA 插件没有在本机编译。

## 预算

| 阶段 | 已测 | 未测完的建议 |
| --- | --- | --- |
| FaceBoxes CPU NMS | 离线编译 3.3s；第二次 G4 功能 PASS | 脚本上限 180s |
| nvdiffrast 纯 Python 包装 | 第一次 G4 2.6s | 上限 180s |
| nvdiffrast CUDA JIT | 2.11 上失败前 39.5s；2.14.1 上还没编过 | 首次建议 180s，缓存目录 `TORCH_EXTENSIONS_DIR` |
| PyTorch3D | 2.14.1 上 **3.893s 编译失败**；更早的 160s 只是超时 | 修好标准后冷编译仍建议 1800s，`MAX_JOBS=4`，超时 124 可续编 |

## 下次在真卡上

不要在这台没有 GPU 的机器上自动重跑。权重 `3DAIGC/LAM-20K` 的 `step_045500` 仍然不在这次范围内。

```bash
git clone https://github.com/zhting/LAM.git
cd LAM
git checkout <本分支>
export LAM_REUSE_SYSTEM_TORCH=1
export TORCH_CUDA_ARCH_LIST=12.0
export NVDIFFRAST_VERBOSE=1
export MAX_JOBS=4
export LAM_COMPAT_CACHE=/content/lam-cache
export TORCH_EXTENSIONS_DIR=/content/lam-cache/torch_extensions
export PYTORCH3D_BUILD_TIMEOUT=1800
bash scripts/compat/bootstrap_blackwell.sh
/content/lam_verify/venv/bin/python scripts/compat/tests/run_cuda_kernel_suite.py \
  --out /content/lam-cache/cuda_kernel_results.json
```

内核套件要用 venv 解释器的绝对路径，不必先 activate。通过标准仍是正面像素大于 0、背面像素为 0。在那之前 CUDA 结果保持 UNTESTED。不要改用 torch 2.3/cu121，也不要在 Python 3.13 上安装根目录 `requirements.txt` 里的 `numpy==1.23.0`。
