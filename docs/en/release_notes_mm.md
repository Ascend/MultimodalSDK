# Release Notes

## Version Information

### Product Version Information

| Item            | Content        |
| --------------- | -------------- |
| Product name    | Multimodal SDK |
| Product version | 26.1.0         |
| Version type    | Release        |

### Related Product Versions

| Product Name | Version |
| ------------ | ------- |
| Ascend HDK   | 26.1.0  |
| CANN         | 9.1.0   |

## Version Compatibility

> [!NOTE]
>
> In the tables in this section, "/" indicates that the versions are incompatible, and "Y" indicates that the versions are compatible.

**Table 1** Multimodal SDK and CANN Version Compatibility

<table style="table-layout: fixed; width: 345px"><colgroup>
<col style="width: 156px">
<col style="width: 91px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Multimodal SDK</th>
    <th colspan="3" style="text-align: center;">CANN Version</th>
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

**Table 2** Multimodal SDK and Ascend HDK Version Compatibility

<table style="table-layout: fixed; width: 345px"><colgroup>
<col style="width: 156px">
<col style="width: 91px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Multimodal SDK</th>
    <th colspan="3" style="text-align: center;">Ascend HDK Version</th>
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

## Important Notes

None

## Update Notes

### New Features

| Feature Name | Feature Description | Supported Product Model |
| -- | -- | -- |
| Keyframe filtering | Adds the `KFrameSelector`/`KRangFrameSelector` classes, which support selecting discrete keyframes related to a query from a video based on text-image similarity. They also support identifying continuous scene intervals and performing adaptive resampling within the intervals. These features are applicable to scenarios such as general video question answering and target occurrence time localization. | Atlas 800I A2 inference server |
| Automatic optimization and multi-scale resampling | Adds an automatic optimization and multi-scale resampling example that provides a reference design for Video RAG-based video understanding and question answering. It supports video frame extraction, audio extraction, ASR, OCR, object detection, semantic retrieval, retrieval reranking, and prompt assembly. It also supports adaptive keyframe selection, uniform sampling fallback, and audio resampling to improve context retrieval and generation for long-video question answering. | Atlas 800I A2 inference server |
| Token compression based on vllm-ascend and Qwen2.5-VL | Adds a Semantic Connected Components (SCC) visual token compression reference design, providing a server-side visual token compression patch for Qwen2.5-VL in vllm-ascend and a verification workflow example. It supports aggregating visual embeddings based on semantic similarity and synchronously adjusting the number of image/video placeholders, reducing the number of visual tokens and improving multimodal inference throughput while preserving model performance as much as possible. | Atlas 800I A2 inference server |

### Service API Changes

**Multimodal SDK**

None

### Key Feature Changes

**Multimodal SDK**

None

### Resolved Issues

None

### Known Issues

None

## Upgrade Impact

### Impact on the Existing System During the Upgrade

None

### Impact on the Existing System After the Upgrade

None

## Documentation for Version 26.1.0

| Document Name | Content Description | Update Notes |
| -- | -- | -- |
| [Multimodal SDK 26.1.0 User Guide](./04_user_guide/user_guide.md) | Provides usage examples and operation guidance for basic preprocessing interfaces in typical image, video, and audio processing scenarios using Multimodal SDK. | For details about the changes, see [Multimodal SDK 26.1.0 User Guide](./04_user_guide/user_guide.md). |

## Virus Scan Results

Virus scan passed.

## Vulnerability Fixes

None
