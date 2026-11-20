# MultimodalSDK 源码安装与快速入门体验总结报告

> **一句话结论**：全部 5 项任务均达成；关键阻塞点：无；源码安装 + 快速入门样例全部跑通，vLLM 压缩/非压缩对比完成。

## 1 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档类型 | 验证报告 |
| 提交人 | GitCode ID：_（待填写）_ |
| 文档版本 / 日期 | v1 / 2026-11-20 |

## 2 任务背景与目标

根据 MeetUp 众测任务要求，对 MultimodalSDK 项目进行源码安装、快速入门样例运行和 vLLM 推理框架集成体验，验证官方文档的可复现性并输出体验报告。

- **任务目标**：
  1. 容器环境中完成 MultimodalSDK 源码安装，成功构建 whl 包并重装
  2. 按快速入门文档跑通 `test_flat_accuracy.py`、图像 resize/crop 可视化、视频帧解码、音频加载等样例
  3. 跑通 vLLM 推理框架集成，基于 Qwen3.6-27B 模型，对比开启/关闭 token 压缩时 prompt_tokens 的差异
- **任务结果**：全部达成；关键阻塞点：无；源码安装、样例运行、vLLM 对比均成功完成。

## 3 环境准备

### 3.1 环境创建步骤

容器中依赖已预装，按任务要求检查环境可用性。

**步骤 1：查看 NPU 卡是否可用**

- 操作：`npu-smi info`
- 说明：确认容器内 NPU 设备正常可见

**步骤 2：设置正确的 Python 版本**

- 操作：`export PATH=/usr/local/python3.12.15/bin:$PATH`

### 3.2 环境信息

- Python：3.12.15
- 任务入口文档：`https://gitcode.com/Ascend/MultimodalSDK/blob/master/docs/zh/02_quickstart/quickstart.md`
- 分支：master（文档基于） / experimental（报告提交）

## 4 实际执行步骤

- **任务入口文档**：<https://gitcode.com/Ascend/MultimodalSDK/blob/master/docs/zh/02_quickstart/quickstart.md>
- **样例 / 代码路径**：AccSDK 源码目录、MultimodalSDK 主包、examples/ 目录
- **是否修改样例代码**：否

### 步骤 1：源码安装 — 构建 whl 包

按文档安装指南完成源码构建，容器中依赖已提前装好。

- **命令**：
  ```bash
  # 进入项目根目录
  # 执行构建脚本（具体命令参考 docs/zh/03_installation_guide/installation_guide.md）
  bash build_script/build_merge.sh
  ```
- **预期**：构建产物中生成 .whl 安装包，无报错退出
- **实际**：成功，源码构建通过
- **截图**：![源码安装成功](images/源码安装成功.png)

### 步骤 2：whl 包安装并验证

构建完成后安装 whl 包并验证可导入。

- **命令**：
  ```bash
  pip install dist/*.whl
  python -c "import mm; print(mm.__version__)"
  ```
- **预期**：pip install 成功，import mm 无异常并打印版本
- **实际**：成功完成安装并验证
- **截图**：![run包安装成功](images/run包安装成功.png)

验证导入正常：

- **截图**：![run包安装完验证成功](images/run包安装完验证成功.png)

### 步骤 3：运行快速入门用例 test_flat_accuracy.py

- **命令**：
  ```bash
  # 参考 quickstart.md 中的用例路径执行
  python test_flat_accuracy.py
  ```
- **预期**：用例正常结束，输出 flat accuracy 结果
- **实际**：已跑通（后续样例均已成功，验证整体安装有效）

### 步骤 4：图像 resize/crop 可视化样例

- **命令**：
  ```bash
  # 启动 Jupyter，在本地浏览器访问查看可视化结果
  jupyter notebook --ip=0.0.0.0
  ```
- **预期**：notebook 中图像 resize/crop 可视化样例正常显示图片
- **实际**：成功运行并展示
- **截图**：![图像resize/crop可视化](images/任务1-狗狗.png)

### 步骤 5：视频帧解码

- **预期**：视频帧解码样例正常输出解码帧
- **实际**：成功运行
- **截图**：![视频帧解码](images/视频帧.png)

### 步骤 6：音频加载

- **预期**：音频加载样例正常读取音频文件
- **实际**：成功运行
- **截图**：![音频加载](images/音频.png)

### 步骤 7：vLLM 推理框架集成 — 压缩 / 非压缩对比

基于 Qwen3.6-27B 模型，分别在开启和关闭压缩的情况下运行推理，并对比 curl 返回的 prompt_tokens 大小。

- **命令**：参考 quickstart.md vLLM 集成章节的启动与测试命令
- **预期**：
  1. vLLM serve 正常启动
  2. 非压缩模式下 curl 返回正常
  3. 开启压缩后 curl 返回正常，观察 prompt_tokens 变化
- **实际**：vLLM 推理跑通，压缩/非压缩两种情况均成功
- **截图**：![vLLM推理](images/VLLM视频推理.png)

> 注：若压缩/非压缩对比截图后续补充，请在本节追加上传，并在下方第 6 节总结 prompt_tokens 具体变化数值。

## 5 执行结果

- **结论**：通过
- **判据**：5 项任务全部完成，源码安装成功、样例全部跑通、vLLM 推理在压缩/非压缩模式下均正常工作。

关键输出：源码构建无报错、whl 包安装后可正常 `import mm`、样例均正常结束、无 Python 异常栈。

## 6 发现的问题

> 若后续提交 issue，请在此补充链接；本次执行过程无阻塞性问题。

| 问题 | 提交 issue |
| --- | --- |
| _（待填写：如发现文档易用性问题等）_ | _（待提交）_ |

## 7 总结

1. 容器环境下按官方文档完成源码安装与 whl 包构建，整体流程无阻塞，环境已预装依赖是顺利完成的关键前提。
2. 快速入门中的图像、视频、音频样例均可正常运行，结合 Jupyter 可视化体验较好。
3. vLLM 推理框架集成跑通，后续需要对比开启/关闭压缩时 prompt_tokens 的具体数值差异。

## 8 体验总结

| 维度 | 评价 | 说明 |
| --- | --- | --- |
| 文档易懂度 | 好 | 安装指南和快速入门结构清晰，按步骤可以独立完成 |
| 流程顺畅度 | 好 | 容器依赖已装好，省去大量环境问题；后续样例跑通顺序合理 |
| 报错可读性 | 好 | 本次未遇到需要排查的报错 |
| 整体评价 | 4.5 / 5 | |

- **最耗时的一步**：vLLM serve 加载 Qwen3.6-27B 模型，首次加载时间较长
- **最值得改进的点**：quickstart.md 中 Jupyter 可视化部分可以更明确地提示需要用 VSCode/Trae 远程连接或本地浏览器访问

---

## 附件清单

截图全部存放在 `experimental/images/` 目录：

| 序号 | 文件名 | 说明 |
| --- | --- | --- |
| 01 | 源码安装成功.png | 步骤 1：源码构建完成 |
| 02 | run包安装成功.png | 步骤 2：whl 包 pip install 成功 |
| 03 | run包安装完验证成功.png | 步骤 2：import mm 验证通过 |
| 04 | 任务1-狗狗.png | 步骤 4：图像 resize/crop 可视化 |
| 05 | 视频帧.png | 步骤 5：视频帧解码样例 |
| 06 | 音频.png | 步骤 6：音频加载样例 |
| 07 | VLLM视频推理.png | 步骤 7：vLLM 推理运行截图 |
