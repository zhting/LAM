# LAM Blackwell 依赖兼容（离线修复）

上游 `aigc3d/LAM` 固定在 `339573649dd93df4cba8093a964e85a80d1b61f3`。改动只在 fork `zhting/LAM`。没有改追踪器、没有假检测器、没有把 import 当成 GPU 通过。静态头像仍然走 FaceBoxes。

## 这次实际跑过什么

本机没有 GPU、没有 nvcc、没有 torch。下面凡是 GPU 的条目都是 **UNTESTED**，不是通过。

| 项目 | G4 实测（RTX PRO 6000 sm_120，Python 3.13.16，torch 2.11.0+cu130） | 本机离线（Python 3.13.18，同一套 NumPy 2.4.6 / Cython 3.0.12 / setuptools 80.10.2） |
| --- | --- | --- |
| CUDA matmul | PASS，0.23s | UNTESTED |
| diff-gaussian-rasterization `8829d14` | 编译 41.4s，前向+反向 PASS | UNTESTED，未重编 |
| simple-knn `60f461f` | 编译 32.7s，核对距离 PASS | UNTESTED，未重编 |
| nvdiffrast `22718580` backface-culling | pip 2.6s，得到 `py3-none-any` wheel；随后 JIT 约 39.5s，`import nvdiffrast_plugin` 失败 | 补丁能打上，纯 Python 包装好；剔除代码仍在。CUDA 插件 **UNTESTED** |
| FaceBoxes CPU NMS | 0.62s 编译失败：`PyArray_Descr` 没有 `subarray` | 同一份旧 `.c` 在这里复现了同一处错误。按当前 NumPy 重新生成后编译 **3.3s**，合成框 NMS **PASS**（0.01s）。GPU NMS 仍 UNTESTED |
| PyTorch3D `e4ad6e5` | 160s 到点，日志停在 “Building wheel started”。不是编译错误 | **UNTESTED**。没有用 CPU 版冒充 |

G4 的 160 秒是诊断脚本的人为上限，不能当成 PyTorch3D 不兼容。

## 三处修复

1. **nvdiffrast**。`setup.py` 本来就不编译 `.so`，`py3-none-any` 是正常的。插件在第一次 `RasterizeCudaContext` 时由 `torch.utils.cpp_extension.load()` 现编。G4 上 `load()` 返回之后，下一行 `importlib.import_module("nvdiffrast_plugin")` 抛了 `ModuleNotFoundError`。torch 2.11 还会在重编时把扩展改名为 `nvdiffrast_plugin_vN`。补丁保留 `load()` 的返回值，不再按未版本化的名字去 import，也不截获编译异常，也不换一个光栅器。`torch_rasterize.cpp` 里的 `RenderModeFlag_EnableBackfaceCulling` 和 GL 路径的 `glEnable(GL_CULL_FACE)` 都没改。显式设置的 `TORCH_CUDA_ARCH_LIST`（例如 `12.0`）不再被清成空字符串。
2. **FaceBoxes**。仓库里的 `cpu_nms.c` 是用 NumPy 1 的 Cython 声明生成的，直接访问已删除的 `PyArray_Descr.subarray`。检出时 `.c` 和 `.pyx` 时间戳相同，Cython 会跳过重新生成，所以 G4 在 0.6 秒内就编了这份旧文件。现在构建会删掉旧 `.c` 并强制从 `.pyx` 生成。`.pyx` 里去掉了 NumPy 2 已删除的 `numpy.int` / `int_t`，索引用 `intp`（与 `argsort` 的 dtype 一致）。没有把失败吞掉再改走 `py_cpu_nms`。
3. **PyTorch3D**。该提交的 `setup.py` 已经带有 CUDA 13 的 `--device-entity-has-hidden-visibility=false` 和 `-static-global-template-stub=false`。单独脚本在持久目录里 `build_ext --inplace`，超时只杀进程、留下 ninja 目标，下次接着编。不要再用 pip 的临时目录，那个目录超时后就没了。

## 时间预算：测过的和估计的

| 阶段 | 测过 | 脚本上限 / 估计 | 说明 |
| --- | --- | --- | --- |
| FaceBoxes CPU NMS | **3.3s** 编译成功 | 脚本上限 180s，只是余量 | `FACEBOXES_BUILD_TIMEOUT` |
| nvdiffrast pip（纯 Python） | G4 **2.6s** | 脚本上限 180s | `NVDIFFRAST_PIP_TIMEOUT` |
| nvdiffrast CUDA JIT | G4 失败前窗口 **39.5s**，成功值未知 | **建议首次 180s**；缓存命中应更短，未测 | 设 `TORCH_EXTENSIONS_DIR`。180s 不是成功实测 |
| GS / simple-knn | G4 **41s / 33s** 且 kernel PASS | 沿用单架构 `12.0`，不必加 PTX | 本机未重编 |
| PyTorch3D | 只测到 **160s 被切断**，成功值未知 | **建议冷编译 1800s**，`MAX_JOBS=4`，只编当前 GPU 的架构 | `PYTORCH3D_BUILD_TIMEOUT`。超时退出码 124，可恢复。31 个 `.cu` + 36 个 `.cpp`。1800s 是估计 |

## 以后在 Colab / G4 上的命令

不要在没有 GPU 的环境里跑下面后半段。权重下载不在本次范围内。checkpoint 是 `3DAIGC/LAM-20K` 的 `step_045500`，直接从 Hugging Face 到运行目录，不要挂 Drive。

```bash
git clone https://github.com/zhting/LAM.git
cd LAM
git checkout <含 scripts/compat 的本分支>
export LAM_REUSE_SYSTEM_TORCH=1
export TORCH_CUDA_ARCH_LIST=12.0
export NVDIFFRAST_VERBOSE=1
export MAX_JOBS=4
export LAM_COMPAT_CACHE=/content/lam-cache
export TORCH_EXTENSIONS_DIR=/content/lam-cache/torch_extensions
export PYTORCH3D_BUILD_TIMEOUT=1800
bash scripts/compat/bootstrap_blackwell.sh
python scripts/compat/tests/run_cuda_kernel_suite.py --out /content/lam-cache/cuda_kernel_results.json
```

确认 JSON 里 `nvdiffrast_backface_culling` 是 `PASS` 且 `back_pixels` 为 0、`front_pixels` 大于 0。在那之前它仍是 UNTESTED。

权重（另一次、需要网络，本次不要自动执行）：

```bash
huggingface-cli download 3DAIGC/LAM-20K \
  --local-dir ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
```

不要 `pip install torch==2.3.0` 的 cu121 轮子，那个构建不覆盖 sm_120。也不要在 Python 3.13 上使用仓库根目录 `requirements.txt` 里的 `numpy==1.23.0`（没有 cp313 wheel）。本栈的 pin 在 `scripts/compat/constraints-blackwell-cu130.txt`。torch 沿用系统里的 `2.11.0+cu130`，不要装进基础环境。

## 风险

- nvdiffrast 的 CUDA 插件和背面剔除还没在真卡上跑通。39 秒只说明编译阶段跑过并在 import 处失败；nvcc 13 / sm_120 仍可能在 `load()` 内部报错。补丁不会把这个错误藏掉。`NVDIFFRAST_VERBOSE=1` 时编译日志会打出来。
- PyTorch3D 冷编译时间没测完。1800 秒是估计。CUDA 13 的 CUB 若没被自动找到，脚本会在存在 `cub/cub.cuh` 时设置 `CUB_HOME`；找不到就会让编译失败，而不是悄悄装一个只有 CPU 的包。
- 离线解释器是 Python 3.13.18，G4 是 3.13.16。ABI 同为 3.13，补丁版本不同。
- xformers、完整 LAM import、FLAME 拟合、照片推理都没跑。FLAME 许可已经处理过，这里没有重新注册。
- FaceBoxes 只编了 CPU NMS。检测器包装本来就只调 `cpu_nms`。
- 旧的 `requirements.txt`（tensorflow 2.12、numpy 1.23 等）和这条 Blackwell 路径不是同一套环境。
