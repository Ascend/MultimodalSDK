# Release Notes

## Key Features

- SCC visual token compression: Aggregates visual embeddings by semantic similarity and adjusts the number of image and video placeholders accordingly. This reduces the number of visual tokens and improves multimodal inference throughput while preserving model accuracy as much as possible.
- Image and video decoding and preprocessing acceleration: The SDK provides high-performance image and video decoding and preprocessing, reducing the preprocessing latency of multimodal inference.
- Keyframe filtering: Filters the discrete keyframes most relevant to the query based on text-image similarity and supports adaptive resampling within contiguous scene intervals, improving context recall for video question answering and target appearance time localization.
- Automatic optimization and multi-scale resampling: Provides a Video RAG-based reference design for video understanding and question answering, covering frame extraction, audio extraction, ASR, OCR, object detection, semantic retrieval, reranking, and prompt assembly, improving context recall and generation quality for long video question answering.

## Version Mapping

### Product Version

| Item | Content |
| -- | -- |
| Product Name | Multimodal SDK |
| Product Version | 26.2.0 |
| Version Type | Release version |

### Related Product Versions

| Product Name | Version |
| -- | -- |
| Ascend HDK | 26.2.0 |
| CANN | 9.2.0 |

## Version Compatibility

> [!NOTE]
>
> In the tables in this section, "/" indicates that the versions cannot be paired and "Y" indicates that they can be paired.

**Table 1**  Multimodal SDK and CANN version compatibility

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
  </tr>
</thead>
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

**Table 2**  Multimodal SDK and Ascend HDK version compatibility

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
| Token compression based on vllm-ascend and Qwen models | Moves the Semantic Connected Components (SCC) visual token compression capability from the reference design into the main code. Therefore, Qwen series VL models in vllm-ascend can use it directly. This feature aggregates visual embeddings by semantic similarity and adjusts the number of image and video placeholders accordingly, reducing the number of visual tokens and improving multimodal inference throughput while preserving model accuracy as much as possible. You configure the compression switch and parameters through environment variables, with no intrusion into existing service code. | Atlas 800I A2 |

### Service Interface Changes

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

### Impact on the System During the Upgrade

None

### Impact on the System After the Upgrade

None

## 26.2.0 Documentation

| Document | Description | Update Notes |
| -- | -- | -- |
| [Multimodal SDK 26.2.0 User Guide](./04_user_guide/user_guide.md) | Mainly includes usage examples and operation guidance for the basic preprocessing interfaces in typical image processing, video processing, and audio processing scenarios of Multimodal SDK. | For details about the changes, see the [Multimodal SDK 26.2.0 User Guide](./04_user_guide/user_guide.md). |

## Virus Scan Results

Virus scan passed.

## Fixed Vulnerabilities

None

## Revision History

**Table 3**

| Document Version | Release Date | Change Description |
| --- | --- | --- |
| 01 | 2026-09-30 | First official release. |
