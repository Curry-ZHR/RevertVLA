# RevertVLA

面向 VLA 推理的闭爪前核验、动作回滚和遮罩重试。[English](README.md)

系统使用 wrist 图像的 SigLIP 判定核验目标商品；未通过时保持姿态和张爪，
完整观察20个控制周期。仍未通过则逆序回放失败手最近成功下发的动作，
另一手保持，再对新的 head 图像做 YOLO 非目标遮罩，交给 VLA 重新推理。

首版规则：**10帧中至少7帧 PASS，且最新两帧 PASS；20周期 HOLD；最近8个
chunk 的成功下发历史；SigLIP362、71类、142条左右手提示词；YOLO v42。**
HOLD 在30Hz约0.667秒，在15Hz约1.333秒；不是固定20张相机图像。

## 只运行核心示例与测试

```bash
git clone https://github.com/Curry-ZHR/RevertVLA.git
cd RevertVLA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pytest
python -m examples.offline_recovery
python -m pytest -q
```

上述流程不加载视觉模型，也不连接或控制机器人。

## 下载权重并校验

模型以 [GitHub Release](https://github.com/Curry-ZHR/RevertVLA/releases/tag/v0.1.1)
附件发布，代码仓库保存配置、类别表、后处理参数和哈希清单。

```bash
python3 scripts/download_weights.py
python3 -B Reverse/check_activation.py
```

SigLIP checkpoint 约1.924GiB，YOLO约59MiB；含下载缓存建议准备至少5GiB空间。
下载器逐附件、逐恢复文件校验 SHA-256。已在浏览器下载附件时，可以传入
`--asset-dir /path/to/assets`。YOLO必须配合仓库中的 `classes.json` 和
`postprocess.json` 使用。旧42类文本仅用于兼容校验，无需旧180步权重。

## Linux / CUDA 推理环境

在执行器所在的同一Linux主机或容器内安装；需要兼容的 NVIDIA 驱动、CUDA12支持和libzstd。

```bash
bash Reverse/setup_runtime.sh
bash Reverse/configure.sh --check-runtime
bash Reverse/configure.sh --warmup-only
```

SigLIP与YOLO使用相互隔离的Python3.12环境。最后一条是显式GPU模型预热诊断，
不订阅机器人相机，也不发布机器人动作。发布时的离线校验不替代目标机器上的GPU和真机验证。

## 接入现有机器人系统

[接入说明](docs/integration.md)列出了相机数据、实际发布回执、停止事件和RTC结果作废接口。
本仓库提供核心与适配协议；VLA服务、机器人SDK、ROS及整站Web应用由宿主系统提供。
只记录经插值、限速后成功下发的目标，不能把未执行预测当作回滚历史。
SDK成功返回也不等于机械臂已经物理到位。

本版保留整幅wrist图像的SigLIP匹配，仍可能受背景商品影响。
夹爪ROI、物体分割，以及VLA训练/权重不在本次发布范围。训练配置、
文件结构、已知边界和许可证详见[英文README](README.md)。
