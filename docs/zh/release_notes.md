# 版本说明

## 版本配套说明

### 产品版本信息

| 项目 | 内容 |
| -- | -- |
| 产品名称 | Multimodal SDK |
| 产品版本 | 26.1.0 |
| 版本类型 | Release 版本 |

### 相关产品版本配套说明

| 产品名称 | 版本 |
| -- | -- |
| Ascend HDK | 26.1.0 |
| CANN | 9.1.0 |

## 版本兼容性说明

> [!NOTE]
>
> 本节表格中“/”表示不可配套，“Y”表示可配套。

**表 1**  Multimodal SDK与CANN版本兼容

<table style="table-layout: fixed; width: 345px"><colgroup>
<col style="width: 156px">
<col style="width: 91px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Multimodal SDK</th>
    <th colspan="3" style="text-align: center;">CANN版本</th>
  </tr>
  <tr>
    <th>9.0.0</th>
    <th>9.1.0</th>
  </tr></thead>
<tbody>
  <tr>
    <td>26.0.0</td>
    <td>Y</td>
    <td>/</td>
  </tr>
  <tr>
    <td>26.1.0</td>
    <td>Y</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

**表 2**  Multimodal SDK与Ascend HDK版本兼容

<table style="table-layout: fixed; width: 345px"><colgroup>
<col style="width: 156px">
<col style="width: 91px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Multimodal SDK</th>
    <th colspan="3" style="text-align: center;">Ascend HDK版本</th>
  </tr>
  <tr>
    <th>26.0.RC1</th>
    <th>26.1.0</th>
  </tr></thead>
<tbody>
  <tr>
    <td>26.0.0</td>
    <td>Y</td>
    <td>/</td>
  </tr>
  <tr>
    <td>26.1.0</td>
    <td>Y</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

## 版本使用注意事项

无

## 更新说明

### 新增特性

| 特性名称 | 特性描述 | 配套产品型号 |
| -- | -- | -- |
| 关键帧筛选 | 新增KFrameSelector/KRangFrameSelector类，支持基于文本-图像相似度从视频中筛选与查询相关的离散关键帧；支持识别连续场景区间并在区间内进行自适应重采样，适用于通用视频问答、目标出现时间定位等场景。 | Atlas 800I A2 推理服务器 |
| AutoOptimizeAndSample | 新增自动寻优与多尺度重采样示例，提供基于 Video RAG 的视频理解与问答参考设计，支持视频帧提取、音频提取、ASR、OCR、目标检测、语义检索、检索重排序和 Prompt 组装，支持自适应关键帧选择、均匀采样回退和音频重采样，提升长视频问答的上下文召回与生成效果。 | Atlas 800I A2 推理服务器 |
| Token 压缩 | 新增 SCC（Semantic Connected Components）视觉 token 压缩参考设计，提供 vllm-ascend 中 Qwen2.5-VL 的服务端视觉 token 压缩补丁和验证流程示例，支持按语义相似度聚合视觉 embedding 并同步调整 image/video placeholder 数量，在尽量保持模型效果的前提下降低视觉 token 数量、提升多模态推理吞吐。 | Atlas 800I A2 推理服务器 |

### 业务接口变更

**Multimodal SDK**

无

### 关键特性变更

**Multimodal SDK**

无

### 已解决的问题

无

### 遗留问题

无

## 升级影响

### 升级过程对现行系统的影响

无

### 升级后对现行系统的影响

无

## 26.1.0 版本配套文档

| 文档名称 | 内容简介 | 更新说明 |
| -- | -- | -- |
| [《Multimodal SDK 26.1.0 用户指南》](./user_guide.md) | 主要包括 Multimodal SDK 图片处理、视频处理和音频处理典型场景的基础预处理接口使用样例与操作指导。 | 变更详见[《Multimodal SDK 26.1.0 用户指南》](./user_guide.md)。 |

## 病毒扫描结果

病毒扫描通过。

## 漏洞修补列表

无
