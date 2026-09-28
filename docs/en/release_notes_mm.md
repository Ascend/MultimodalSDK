# Release Notes

## Key Features

- SCC visual token compression : The SDK aggregates visual embeddings by semantic similarity and adjusts the number of image/video placeholders accordingly, reducing visual token count and improving multimodal inference throughput while preserving model accuracy as much as possible.
- Accelerated image and video decoding and preprocessing : The SDK delivers high-performance decoding and preprocessing for images and videos, reducing preprocessing latency for multimodal inference workloads.
- Key frame selection : The SDK selects the discrete key frames most relevant to a given text query based on text-image similarity, and supports adaptive resampling within continuous scene intervals, improving context recall for video question answering and target occurrence localization.
- Automatic optimization and multi-scale resampling : The SDK provides a Video RAG-based reference design for video understanding and question answering, covering frame extraction, audio extraction, ASR, OCR, object detection, semantic retrieval, re-ranking, and prompt assembly, which improves context recall and generation quality for long video question answering.

## Version Information

### Product Version Information

| Item            | Content        |
| --------------- | -------------- |
| Product name    | Multimodal SDK |
| Product version | 26.2.0         |
| Version type    | Release        |

### Related Product Versions

| Product Name | Version |
| ------------ | ------- |
| Ascend HDK   | 26.2.0  |
| CANN         | 9.2.0   |

## Version Compatibility

> [!NOTE]
>
> In the tables in this section, "/" indicates that the versions are incompatible, and "Y" indicates that the versions are compatible.

**Table 1** Multimodal SDK and CANN Version Compatibility

<table style="table-layout: fixed; width: 345px"><colgroup>
<col style="width: 156px">
<col style="width: 98px">
<col style="width: 98px">
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
    <th>9.2.0</th>
  </tr></thead>
<tbody>
  <tr>
    <td>26.0.0</td>
    <td>Y</td>
    <td>/</td>
    <td>/</td>
  </tr>
  <tr>
    <td>26.1.0</td>
    <td>Y</td>
    <td>Y</td>
    <td>/</td>
  </tr>
  <tr>
    <td>26.2.0</td>
    <td>Y</td>
    <td>Y</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

**Table 2** Multimodal SDK and Ascend HDK Version Compatibility

<table style="table-layout: fixed; width: 345px"><colgroup>
<col style="width: 156px">
<col style="width: 98px">
<col style="width: 98px">
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
    <th>26.2.0</th>
  </tr></thead>
<tbody>
  <tr>
    <td>26.0.0</td>
    <td>Y</td>
    <td>/</td>
    <td>/</td>
  </tr>
  <tr>
    <td>26.1.0</td>
    <td>Y</td>
    <td>Y</td>
    <td>/</td>
  </tr>
  <tr>
    <td>26.2.0</td>
    <td>Y</td>
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
| Token Compression for vllm-ascend with Qwen Models | Integrate the SCC (Semantic Connected Components) visual token compression capability from the reference design into the mainline code, making it directly available for Qwen-series VL models in vllm-ascend. This feature aggregates visual embeddings by semantic similarity and synchronously adjusts image/video placeholder counts, reducing the number of visual tokens and improving multimodal inference throughput while preserving model accuracy to the extent possible. The compression switch and parameters are configured via environment variables, with zero intrusion into existing business code. | Atlas 800I A2 inference server |

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

## Documentation for Version 26.2.0

| Document Name | Content Description | Update Notes |
| -- | -- | -- |
| [Multimodal SDK 26.2.0 User Guide](./04_user_guide/user_guide.md) | Provides usage examples and operation guidance for basic preprocessing interfaces in typical image, video, and audio processing scenarios using Multimodal SDK. | For details about the changes, see [Multimodal SDK 26.2.0 User Guide](./04_user_guide/user_guide.md). |

## Virus Scan Results

Virus scan passed.

## Vulnerability Fixes

None

## Revision History

**Table 3**

| Document Version | Release Date | Description of Change |
| --- | --- | --- |
| 01 | 2026-09-30 | First official release. |
