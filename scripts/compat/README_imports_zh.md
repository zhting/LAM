# 无权重的完整导入检查

这是给下一次有限 G4 用的准备，不是推理结果。头 `0b55127` 在 Python 3.13.16、torch 2.14.1+cu130、CUDA 13、NumPy 2.4.6、RTX PRO 6000 sm_120 上再次通过六个依赖内核。PyTorch3D 编译 241.187 秒，冷引导 529.048 秒，GS 35.070 秒，simple-knn 29.311 秒。导入安装在 10.384 秒因 `decorator==5.1.1` 与 `moviepy==1.0.3` 冲突失败。`ModelLAM`、`FlameTrackingSingleImage`、`GlobalTracker`、runner 导入，以及 torchvision / xformers 算子都还没有执行。

## 导入链里真正会碰到的东西

- `lam.models.modeling_lam.ModelLAM` 在导入时拉起 GS 渲染器、PyTorch3D、diff-gaussian。构造函数会建 DINO 编码器，`pretrained=True` 会从 `dl.fbaipublicfiles.com` 下载权重。本阶段只导入类。
- `tools.flame_tracking_single_image.FlameTrackingSingleImage` 的导入会带上 FaceBoxes、VGGHead、StyleMatte、`vhap.model.tracker.GlobalTracker`。这些类的构造函数会 `torch.load` 或读取 FLAME pkl。本阶段不调用构造函数，记为 `BLOCKED_RESOURCE`。
- `external.human_matting.stylematte` 在导入时需要 `transformers` 的 Mask2Former。用的是本地类，不下载权重。
- `lam.runners.infer.lam` 额外导入 `mcubes` 和 `moviepy.editor`。moviepy 必须是 1.0.3，2.x 没有 `moviepy.editor`。它要求 `decorator>=4.0.2,<5`，所以钉 `decorator==4.4.2`，不能用 5.1.1。
- `diffusers==0.32.2` 才和 `transformers==4.46.3`、`huggingface-hub==0.26.5`、`safetensors==0.4.5` 同时成立。0.42.1 要 hub>=1.32 且 safetensors>=0.8，和这条 transformers 线冲突。0.32.2 仍提供 `diffusers.utils.is_torch_version`。
- torchvision 被追踪器和 matting 直接使用。与 torch 2.14.1+cu130 对应的轮子是 cu130 索引上的 `torchvision==0.29.1+cu130`（cp313 manylinux）。不要装 PyPI 上那个会声明 `torch>=2.14` 的元数据轮子，安装时加 `--no-deps`。
- xformers 只在 DINOv2 的 `MemEffAttention` 里可选。上游已经用 `ImportError` 退回普通注意力；`attn_bias` 不是 None 时才强制需要它。cu130 索引有 `xformers==0.0.35`（标签 py39-none，`_C.so` 链接 libtorch_cuda / libcudart.so.13，不链接 libpython）。没有 cp313 专用轮子，也不要从源码编。装不上或导入失败时，算子记 UNTESTED/FAIL，不要改 LAM 把回退伪装成算子通过。
- TensorFlow 2.12 和 face-detection-tflite 没有 cp313 轮子。虹膜检测是 `fdlite` 的函数内导入，默认追踪器导入不会走到。不安装。chumpy 只在反序列化 FLAME 时需要，本阶段不碰。

## 预算

| 阶段 | 已测 | 这次的上限 / 估计 |
| --- | --- | --- |
| 已有内核引导 | 531s + GS 37s + knn 30s + 套件 1.6s | 不要重编，除非 venv 不在了 |
| PyTorch3D | 241.875s 成功 | 已含在上面的引导里 |
| 纯 Python 依赖 + torchvision/xformers 轮子 | 未测完。两个 CUDA 轮子约 7MB 和 5MB | `IMPORT_STAGE_TIMEOUT` 默认 600s。估计 2–4 分钟，不是实测 |
| pymcubes 0.1.6 | 无 cp313 轮子。本机 clang 缺 `<cmath>`，不能当成 G4 失败 | 上限 120s。G4 的 gcc 可能编过 |
| 导入冒烟 | 本机无 torch，未执行 | 上限约 3 分钟。估计远小于此 |

缓存还在时，这一段加上冒烟估计在 15 分钟内，落在 40 分钟里面。冷启动若要重跑内核引导，用已经测过的约 9 分钟，再加这一段。

## 分配 GPU 之前可以跑的预检

用本机 Python 3.13 即可，不需要 torch，也不会改系统 torch。它会复现 `decorator==5.1.1` 与 `moviepy==1.0.3` 的 `ResolutionImpossible`，再对当前约束做 `--dry-run`，并核对 torchvision / xformers / accelerate / diffusers 的依赖是否被这些 pin 加上已证明的 `torch==2.14.1+cu130` 满足。

```bash
python3 scripts/compat/preflight_imports.py --report /tmp/import_preflight.json
```

## 给父进程的完整顺序

不要 `pip install torch`。不要下载 LAM-20K 或 FLAME。导入冒烟仍不是推理。torchvision CUDA NMS 和 xformers 算子只有在这次 G4 上跑过才算数；写这份说明时它们是 **UNTESTED**。

已测阶段限时来自头 `0b55127` 之前那次成功内核和这次失败的导入安装：

| 顺序 | 命令 | 已测 | 上限 |
| --- | --- | --- | --- |
| 0 | `preflight_imports.py` | 离线 dry-run，见结果 JSON | 无 GPU |
| 1 | `bootstrap_blackwell.sh` | 529.048s，含 PyTorch3D 241.187s | PyTorch3D 1800s |
| 2 | GS / simple-knn | 35.070s / 29.311s | 沿用现有构建 |
| 3 | `run_cuda_kernel_suite.py` | 六项 PASS | 约 2 分钟 |
| 4 | `bootstrap_imports.sh` | 上次 10.384s 在解析阶段失败 | `IMPORT_STAGE_TIMEOUT` 600s |
| 5 | `smoke_full_imports.py` | 未执行 | 估计 <180s |

```bash
python3 scripts/compat/preflight_imports.py --report /tmp/import_preflight.json
cd /path/to/LAM
git fetch origin cursor/lam-blackwell-deps-9075
git checkout cursor/lam-blackwell-deps-9075
export LAM_VENV=/content/lam_verify/venv
export LAM_COMPAT_CACHE=/content/lam_verify/cache
export LAM_REPORT_DIR=/content/lam_verify/report
bash scripts/compat/bootstrap_blackwell.sh
# GS 与 simple-knn 若该 venv 里还没有，再按上次同样的 git pin 安装
"$LAM_VENV/bin/python" scripts/compat/tests/run_cuda_kernel_suite.py \
  --out "$LAM_REPORT_DIR/cuda_kernel_results.json"
export IMPORT_STAGE_TIMEOUT=600
export PYMCUBES_TIMEOUT=120
bash scripts/compat/bootstrap_imports.sh
"$LAM_VENV/bin/python" scripts/compat/tests/smoke_full_imports.py \
  --out "$LAM_REPORT_DIR/smoke_full_imports.json"
```

## 给父进程的命令（仅导入段，内核 venv 已在）

不要在没有 GPU 的机器上启动。不要 `pip install torch`。不要下载 LAM-20K 或 FLAME。

```bash
cd /path/to/LAM
git fetch origin cursor/lam-blackwell-deps-9075
git checkout cursor/lam-blackwell-deps-9075
export LAM_VENV=/content/lam_verify/venv
export LAM_COMPAT_CACHE=/content/lam_verify/cache
export LAM_REPORT_DIR=/content/lam_verify/report
export IMPORT_STAGE_TIMEOUT=600
export PYMCUBES_TIMEOUT=120
bash scripts/compat/bootstrap_imports.sh
"$LAM_VENV/bin/python" scripts/compat/tests/smoke_full_imports.py \
  --out "$LAM_REPORT_DIR/smoke_full_imports.json"
```

`LAM_VENV` 不存在时脚本会马上失败，并指向已经测过的 `scripts/compat/bootstrap_blackwell.sh`。torch 不是 `2.14.1+cu130` 或 CUDA 不是 13.0 时也会马上失败，不会换一个 torch。

冒烟 JSON 的 `overall: PASS` 只表示导入和列出的合成算子。`blocked_resource` 里的构造、拟合、推理都没有跑。没有 CUDA 时 torchvision CUDA NMS 和 xformers 算子是 `UNTESTED`。
