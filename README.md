<h1 align="center">Multimodal SDK</h1>

<div align="center">
<h2>昇腾多模态开发套件</h2>

[![Ascend](https://img.shields.io/badge/Community-MindSDK-blue.svg)](https://www.hiascend.com/cn/developer/software/mindsdk)
[![License](https://badgen.net/badge/License/MulanPSL-2.0/blue)](./LICENSE.md)
[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/MultimodalSDK)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/MultimodalSDK)

</div>

## ✨ 最新消息

- **[2026.04.25]**：[Multimodal SDK 26.0.0](https://gitcode.com/Ascend/MultimodalSDK/releases/v26.0.0) 发布
- **[2025.12.30]**：Multimodal SDK 开源发布

## ℹ️ 简介

多模态大模型推理流程中需要处理大量复杂的数据。Multimodal SDK 通过提供一系列高性能的昇腾设备亲和性接口，加速大模型推理预处理流程，包括图像视频加载和解码、resize、crop 等预处理常用操作，并支持多种开源数据结构与加速库数据结构的相互转换，方便快速使用和移植。

![](./docs/zh/figures/mmsdk_arch.svg)

## ⚙️ 功能介绍

| 类别 | 模块 | 功能简介 | 文档 |
|:--:|:--|:--|:--|
| 加速库 | 功能函数 | Tensor / Image / video_decode / load_audio 等预处理接口 | [功能函数参考](docs/zh/05_api/function_reference.md) |
| 适配器 | Adapter | Qwen2VL、InternVL2 模型预处理适配 | [Adapter](docs/zh/05_api/adapter.md) |
| 补丁 | Patcher | vLLM 框架预处理加速补丁 | [patcher](docs/zh/05_api/patcher.md) |
| API | Python 接口 | 数据类型枚举与接口目录 | [Python 接口说明](docs/zh/05_api/README.md) |

## 🚀 快速入门

Multimodal SDK提供了一个简单的样例，帮助用户通过 Docker 快速启动环境并完成首次体验，详见[快速入门](docs/zh/02_quickstart/quickstart.md)。

## 📦 安装指南

详细的安装部署说明请参见[安装指南](docs/zh/03_installation_guide/installation_guide.md)。

## 📘 使用指南

详细的开发者文档请参见[Multimodal SDK 开发者文档](docs/zh/README.md)。

## 🛠️ 贡献指南

欢迎参与项目贡献，请参见[贡献指南](CONTRIBUTING.md)。

## ⚖️ 相关说明

- [许可证声明](LICENSE.md)（docs 文档适用 CC-BY 4.0，见 [LICENSE](docs/LICENSE)）
- [免责声明](docs/zh/01_introduction/02_disclaimer.md)

## 🤝 建议与交流

欢迎大家通过以下方式提出问题、交流讨论。

| 资源 | 说明 |
|:--|:--|
| [FAQ](./docs/zh/06_references/faq.md) | 常见问题解答与使用答疑 |
| [创建Issue](https://gitcode.com/Ascend/MultimodalSDK/issues/create/choose) | 提交 Bug、需求或建议 |
| [社区任务](https://gitcode.com/Ascend/MultimodalSDK/issues) | 查看和认领社区任务 |
| [会议日历](https://meeting.ascend.osinfra.cn/?sig=sig-MindSeriesSDK) | 社区定期例会与活动日程 |
