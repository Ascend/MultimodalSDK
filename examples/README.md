# 1 目录结构

```text
examples/
|-- README.md                            // 说明文档
|-- common.py                             // 公共工具模块（图像编码、消息构建、视频解码、VLM调用等）
|-- k_frame_selector_example.py           // 离散关键帧问答示例，基于 KFrameSelector + SimpleQaDemo
`-- k_range_frame_selector_example.py     // 区间关键帧问答示例，基于 KRangFrameSelector + RangeDetectionQaDemo
```

## 2 接口描述及使用实例

### 2.1 SimpleQaDemo（k_frame_selector_example.py）

离散关键帧问答。基于 `KFrameSelector`，从视频中选择与查询相关的离散关键帧，送入大模型获取回答。适用于通用视频问答场景。

```python
class SimpleQaDemo:

    def __init__(self,
                 model_path: str,
                 device_list: list,
                 similar_threshold: float = 0.06,
                 similar_threshold_image: float = 0.015,
                 vlm_url: str = None,
                 api_key: str = "NONE",
                 vlm_model_name: str = None,
                 model_type: str = 'cn_clip'
                 ):
        """
        :param model_path: 模型路径
        :param device_list: device列表
        :param similar_threshold: 问题，视频帧相似度的最大差值
        :param similar_threshold_image: 图片相似度的最大差值
        :param vlm_url: VLLM模型的推理URL
        :param api_key: API密钥
        :param vlm_model_name: 推理服务的模型名称
        :param model_type: 加载的clip模型类型，取值范围 clip, cn_clip
        """

    def qa(self, query: str, video_path: str, sample_num: int):
        """
        :param query: 问题
        :param video_path: 视频路径
        :param sample_num: 最大采样帧数
        :return: 大模型回答文本
        """
```

#### 使用实例

```python
from k_frame_selector_example import SimpleQaDemo

demo = SimpleQaDemo(
    model_path="/path/to/chinese-clip-vit-large-patch14-336px",
    device_list=[0],
    vlm_url="http://your-vlm-url/v1",
    api_key="None",
    vlm_model_name="Qwen2.5-VL-32B-Instruct",
    model_type='cn_clip'
)

answer = demo.qa("视频中出现了哪些交通标志", "/path/to/video.mp4", sample_num=4)
```

### 2.2 RangeDetectionQaDemo（k_range_frame_selector_example.py）

区间关键帧问答。基于 `KRangFrameSelector`，从视频中定位与查询相关的连续区间关键帧，送入大模型获取时间区间回答。适用于需要时间定位的场景（如"红色小轿车什么时候出现的"）。

内部通过 VLM 对用户查询进行重写，自动选择采样策略（基于关键词的关键帧采样或均匀采样），再调用对应的关键帧选择器。

```python
class RangeDetectionQaDemo:

    def __init__(self,
                 model_path: str,
                 device_list: list,
                 similar_threshold: float = 0.03,
                 similar_threshold_image: float = 0.015,
                 vlm_url: str = None,
                 api_key: str = "NONE",
                 vlm_model_name: str = None,
                 model_type: str = 'cn_clip'
                 ):
        """
        :param model_path: 模型路径
        :param device_list: device列表
        :param similar_threshold: 问题，视频帧相似度的最大差值(即和最相似的那帧之间的差值的最大范围)
        :param similar_threshold_image: 图片相似度的最大差值
        :param vlm_url: VLLM模型的推理URL
        :param api_key: API密钥
        :param vlm_model_name: 推理服务的模型名称
        :param model_type: 加载的clip模型类型，取值范围 clip, cn_clip
        """

    def qa(self, query: str, video_path: str, sample_num: int):
        """
        :param query: 问题
        :param video_path: 视频路径
        :param sample_num: 最大采样帧数
        :return: 大模型回答文本（JSON格式的时间区间列表）
        """
```

#### 使用实例

```python
from k_range_frame_selector_example import RangeDetectionQaDemo

demo = RangeDetectionQaDemo(
    model_path="/path/to/chinese-clip-vit-large-patch14-336px",
    device_list=[4],
    similar_threshold=0.03,
    similar_threshold_image=0.015,
    vlm_url="http://your-vlm-url/v1",
    api_key="None",
    vlm_model_name="Qwen2.5-VL-32B-Instruct",
    model_type='cn_clip'
)

answer = demo.qa("白色公交车是什么时候出现和消失的？", "/path/to/video.mp4", sample_num=16)
```

## 3 执行用例

### 3.1 启动vllm推理服务

```bash
# 启动镜像
docker run -itd --shm-size=512g --name keyframe_test --device=/dev/davinci0:/dev/davinci0 --device=/dev/davinci1:/dev/davinci1 --device=/dev/davinci2:/dev/davinci2 --device=/dev/davinci3:/dev/davinci3 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc --network=host -v /usr/local/dcmi:/usr/local/dcmi  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -v /root/.cache:/root/.cache -v /home:/home quay.io/ascend/vllm-ascend:v0.15.0rc1

# 进入镜像
docker exec -it keyframe_test /bin/bash

# 启动vllm server
vllm serve /data/models/Qwen2.5-VL-32B-Instruct/ -tp=4 --max-model-len=100000 --served-model-name=Qwen2.5-VL-32B-Instruct --mm_processor_cache_gb=0

```

### 3.2 安装Multimodal SDK及依赖

```bash
pip install transformers==4.51.3 "pillow>=11.2.1" numpy==1.26.4 opencv-python decord2 qwen_vl_utils openai einops accelerate decorator scipy attrs
wget https://gitcode.com/Ascend/MultimodalSDK/releases/download/v26.1.0/Ascend-mindxsdk-multimodal_26.1.0_linux-aarch64.run
bash Ascend-mindxsdk-multimodal_26.1.0_linux-aarch64.run --install --install-path=/usr/local/
source /usr/local/multimodal/script/set_env.sh
```

> [!NOTE] 说明
>
> `torchvision` 不是本样例直接依赖的业务组件，因此上述命令不单独安装 `torchvision`。如果需要使用 `torchvision`，其版本必须与环境中的 `torch` 配套，不能直接安装未指定版本的最新版。例如，`torch==2.9.1` 对应使用 `torchvision==0.24.1`。

### 3.3 执行测试样例

根据实际场景修改k_frame_selector_example.py及k_range_frame_selector_example.py中的参数

```python
model_path="/path/to/chinese-clip-vit-large-patch14-336px",
device_list=[0],
vlm_url="http://127.0.0.1:8111/v1",
api_key="None",
vlm_model_name="Qwen2.5-VL-32B-Instruct",
model_type='cn_clip',
```

执行测试文件：

```bash
python k_frame_selector_example.py
python k_range_frame_selector_example.py
```

运行后终端将输出形如下方的结果，说明运行成功。

```text
查询: 红色小轿车是什么时候出现和消失的
回答: [
    {
        "start_time": "00:40",
        "end_time": "00:52"
    }
]
```
