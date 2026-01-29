# Adapter<a name="ZH-CN_TOPIC_0000002423351984"></a>

## MultimodalQwen2VLImageProcessor<a name="ZH-CN_TOPIC_0000002466418117"></a>

该函数继承transformers库中的Qwen2VLImageProcessor，将使用多模态内部的加速能力对Qwen2VL模型的图像/视频预处理环节进行加速，返回与transformers一致的BatchFeature类型。

**使用基本说明<a name="section220615283319"></a>**

对于图像/视频数据的预处理，多模态当前仅支持对接transformers4.51.3版本的处理能力。

### \_\_init\_\_<a name="ZH-CN_TOPIC_0000002432982122"></a>

**功能描述<a name="section184861730122811"></a>**

类初始化函数。

**函数原型<a name="section189169235416"></a>**

```
def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: Resampling = Resampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = RESCALE_FACTOR,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: Optional[int] = 56 * 56,
        max_pixels: Optional[int] = 28 *28 *1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
) -> None:
```

**参数说明<a name="section1320011141163"></a>**

|参数名|类型|说明|可选/必选|当前版本是否支持设置|
|--|--|--|--|--|
|do_resize|bool|是否对输入图像进行缩放，默认为True。|可选|×|
|size|Dict[str, int]|输入图像的最小和最大尺寸，格式必须为{"shortest_edge": min_pixels, "longest_edge": max_pixels}，默认为None。|可选|√|
|resample|Resampling|图像缩放插值方式，默认为Resampling.BICUBIC。|可选|×|
|do_rescale|bool|是否执行rescale（缩放像素值到特定范围），默认为True。|可选|×|
|rescale_factor|Union[int, float]|像素缩放因子，通常为1/255，默认为RESCALE_FACTOR。|可选|×|
|do_normalize|bool|是否对图像执行标准化，默认为True。|可选|×|
|image_mean|Optional[Union[float, List[float]]]|图像标准化均值，若为None，则使用默认值。默认值为[0.48145466, 0.4578275, 0.40821073]。|可选|√|
|image_std|Optional[Union[float, List[float]]]|图像标准化方差，若为None，则使用默认值。默认值为[0.26862954, 0.26130258, 0.27577711]。|可选|√|
|do_convert_rgb|bool|是否将输入图像转换为RGB，默认为True。|可选|×|
|min_pixels|Optional[int]|输入图像的最小像素数。默认为56*56。|可选|√|
|max_pixels|Optional[int]|输入图像的最大像素数。高于该值时会触发降采样。默认为28*28*1280。|可选|√|
|patch_size|int|空间维度切分patch的大小（像素），默认为14。|可选|√|
|temporal_patch_size|int|时间维度切分patch的大小（帧），默认为2，且不支持设置。|可选|×|
|merge_size|int|patch合并时的大小，默认为2。|可选|√|
|kwargs|dict|其他扩展参数。|可选|×|


>[!CAUTION] 注意 
>初始化本预处理模块时，以下参数均可以传入值，但在此版本中具有以下限制：
>-  **固定启用参数**（不可配置）：
>    -   do\_resize：该参数始终启用，不支持关闭。
>    -   do\_rescale：该参数始终启用，不支持关闭。
>    -   do\_normalize：该参数始终启用，不支持关闭。
>-  **不支持的参数**：
>    -   do\_convert\_rgb：不支持该参数，仅允许输入RGB图像。
>    -   data\_format：不支持该参数，输出数据格式固定为通道优先\(C,H,W\)。
>    -   resample：不支持该参数，插值方式固定为双三次插值\(BICUBIC\)。
>    -   rescale\_factor：不支持该参数，缩放因子固定为1/255。
>    -   input\_data\_format：不支持该参数，仅支持RGB格式，通道顺序为HWC。
>    -   kwargs: 其他扩展参数。

**示例<a name="section1587174015349"></a>**

```
from mm import MultimodalQwen2VLImageProcessor
processor = MultimodalQwen2VLImageProcessor(
min_pixels=3136,
max_pixels=518400,
patch_size=14,
temporal_patch_size=2,
merge_size=2,
image_mean=[0.48145466, 0.4578275, 0.40821073],
image_std=[0.26862954, 0.26130258, 0.27577711]
)
```


### preprocess<a name="ZH-CN_TOPIC_0000002466501301"></a>

**功能描述<a name="section184861730122811"></a>**

预处理函数，对输入的图像/视频，根据特定的超参数进行预处理，以BatchFeature的通用形式返回。

**函数原型<a name="section719310341377"></a>**

```
def preprocess(self,
               images: ImageInput,
               videos: VideoInput = None,
               do_resize: Optional[bool] = None,
               size: Dict[str, int] = None,
               min_pixels: Optional[int] = None,
               max_pixels: Optional[int] = None,
               resample: Resampling = None,
               do_rescale: Optional[bool] = None,
               rescale_factor: Optional[float] = None,
               do_normalize: Optional[bool] = None,
               image_mean: Optional[Union[float, List[float]]] = None,
               image_std: Optional[Union[float, List[float]]] = None,
               patch_size: Optional[int] = None,
               temporal_patch_size: Optional[int] = None,
               merge_size: Optional[int] = None,
               do_convert_rgb: Optional[bool] = None,
               return_tensors: Optional[str] = None,
               data_format: Optional[str] = None,
               input_data_format: Optional[str] = None) -> BatchFeature
```

**参数说明<a name="section1952764818252"></a>**

|参数名|类型|说明|可选/必选|当前版本是否支持设置|
|--|--|--|--|--|
|images|ImageInput|ImageInput支持如下内容：<p>**单张图像**：</p><ul><li>Image：多模态的图像对象。</li><li>PIL.Image.Image：PIL图像对象。</li><li>np.ndarray：Numpy格式图像，支持HWC（高度 × 宽度 × 通道）。</li></ul><p>**多张图像**：</p>List[...]：上述任意类型的列表，表示多张图像。<p>所有图像大小限制为10\*10-4096\*4096且满足宽和高均大于patch_size * merge_size。</p>|必选|√|
|videos|Optional[VideoInput]|VideoInput默认为None，支持输入如下内容：<ul><li>**单帧视频**：单张图像为1帧视频。</li><li>**单个视频序列**：SingleVideoInput。</li><li>**多个视频**：List[SingleVideoInput]，每个元素是一个视频。</li></ul>其中单帧图像，支持Numpy、PIL或多模态Image对象。<p>单个视频的输入支持以下形式：</p><ul><li>np.ndarray：形状[T, H, W, C]的4维张量，表示T帧视频。</li><li>List[FrameInput]：逐帧组成的列表。</li></ul>视频帧的大小限制为10\*10-4096\*4096且满足宽和高均大于patch_size * merge_size。|可选|√|
|do_resize|Optional[bool]|是否对图像进行缩放。该参数不可配置。|可选|×|
|size|Dict[str, int]|输入图像的最小和最大尺寸。格式必须为{"shortest_edge": min_pixels, "longest_edge": max_pixels}，默认为None。输入的min_pixels与max_pixels应满足下方min_pixels和max_pixels的约束。|可选|√|
|min_pixels|Optional[int]|输入图像的最小像素数，低于该值时可能报错或拒绝。min_pixels取值范围为[10*10, max_pixels)。|可选|√|
|max_pixels|Optional[int]|输入图像的最大像素数，高于该值时会触发降采样。max_pixels取值范围为(min_pixels, 4096*4096]。|可选|√|
|resample|Optional[Resampling]|插值方式固定为BICUBIC。|可选|×|
|do_rescale|Optional[bool]|是否执行像素值缩放。|可选|×|
|rescale_factor|Optional[float]|缩放因子固定为1/255。|可选|×|
|do_normalize|Optional[bool]|是否执行归一化。|可选|×|
|image_mean|Optional[Union[float, List[float32]]]|标准化均值，若为None，则使用初始化值，取值范围为[0, 1]。|可选|√|
|image_std|Optional[Union[float32, List[float32]]]|标准化方差，若为None，则使用初始化值，取值范围在(0, FLOAT32_MAX)。|可选|√|
|patch_size|Optional[int]|空间维度切分patch的大小（像素），若为None，则使用初始化值。|可选|√|
|temporal_patch_size|Optional[int]|时间维度切分patch的大小（帧），默认为2。|可选|×|
|merge_size|Optional[int]|patch合并时的大小，若为None，则使用初始化值。|可选|√|
|do_convert_rgb|Optional[bool]|输入必须为RGB图像。|可选|×|
|return_tensors|Optional[str]|返回张量格式，支持"np"、"pt" 等。|可选|√|
|data_format|Optional[str]|输出数据格式固定为通道优先 (C,H,W)。|可选|×|
|input_data_format|Optional[str]|输入数据格式固定为RGB且为HWC排布。|可选|×|


>[!CAUTION] 注意 
>使用本函数时，无论是否在初始化中初始化过以下参数，其均具有以下约束：
>-  **固定启用参数**（不可配置）：
>    -   do\_resize：该参数始终启用，不支持关闭。
>    -   do\_rescale：该参数始终启用，不支持关闭。
>    -   do\_normalize：该参数始终启用，不支持关闭。
>-  **不支持的参数**：
>    -   do\_convert\_rgb：不支持该参数，仅允许输入RGB图像。
>    -   data\_format：不支持该参数，输出数据格式固定为通道优先 \(C,H,W\)。
>    -   resample：不支持该参数，插值方式固定为双三次插值 \(BICUBIC\)。
>    -   rescale\_factor：不支持该参数，缩放因子固定为1/255。
>    -   input\_data\_format：不支持该参数，仅支持RGB格式，通道顺序为HWC。
>-  **使用时还需满足的额外约束如下**：
>    -   输入图像或视频帧为U8类型RGB图像，数据排布限制为HWC，大小限制为10\*10-4096\*4096。
>    -   输入min\_pixels范围为\[10\*10, max\_pixels\)。
>    -   输入max\_pixels范围为\(min\_pixels, 4096\*4096\]。
>    -   输入Image的宽和高均大于patch\_size \* merge\_size。
>    -   对于传入的每一张图像，或者每一个视频帧，若其宽高分别为w, h，则需满足min\_pixels < max\_pixels, max\_pixels \>= w/h\*\(patch\_size \* merge\_size\)^2，patch\_size \* merge\_size <= h，w。
>    -   对于传入的视频，每一个视频帧的宽、高及数据排布格式需一致。

**示例<a name="section14194123415712"></a>**

```
from mm import MultimodalQwen2VLImageProcessor
import numpy as np
processor = MultimodalQwen2VLImageProcessor(min
min_pixels=3136,
max_pixels=518400,
patch_size=14,
temporal_patch_size=2,
merge_size=2,
image_mean=[0.48145466, 0.4578275, 0.40821073],
image_std=[0.26862954, 0.26130258, 0.27577711]
)

def random_image_array():
    h = 1024
    w = 1024
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return arr

def random_video_list(num_frames=None):
    num_frames = num_frames or np.random.randint(2, 5)
    return [random_image_array() for _ in range(num_frames)]
video = [random_video_list()]
result = processor.preprocess(images=[], videos=video)
```



## InternVL2PreProcessor<a name="ZH-CN_TOPIC_0000002435238562"></a>

### preprocess\_image<a name="ZH-CN_TOPIC_0000002436004904"></a>

**功能描述<a name="section184861730122811"></a>**

预处理函数，使用InternVL预处理流程处理输入的图像。为InternVL2PreProcessor的成员函数，使用时需要先初始化InternVL2PreProcessor对象。

**函数原型<a name="section719310341377"></a>**

```
def preprocess_image(
            image: Union[PIL.Image.Image, Image],
            input_size: int,
            min_num: int,
            max_num: int,
            use_thumbnail: bool,
    ) -> torch.Tensor:
```

**参数说明<a name="section1952764818252"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|image|[PILImage.Image, Image]|必选|Image支持如下内容：<p>**单张图像**：</p><ul><li>Image：多模态的图像对象。</li><li>PIL.Image.Image：PIL图像对象。</li></ul>|
|input_size|int|必选|InternVL处理流程中每个被crop的图像缩放大小（限制为10-8192）。|
|min_num|int|必选|用于计算目标缩放比例的最小数量。限制取值范围为[1, 4]且小于max_num。|
|max_num|int|必选|用于计算目标缩放比例的最大数量。限制取值范围为(min_num,32]。|
|use_thumbnail|bool|必选|是否加入原图的缩略图。|


>[!CAUTION] 注意 
>使用本函数时请注意对于input\_size缩放比的限制为8192，但由于缩放比例计算的问题，最终计算出需要将原图缩放的大小可能会超过8192，此时会被底层接口拦截报错。

**示例<a name="section14194123415712"></a>**

```
from mm import Image, InternVL2PreProcessor

image = Image.open("/home/test.jpeg", "cpu")

internVL2PreProcessor = InternVL2PreProcessor()
result = internVL2PreProcessor.preprocess_image(image, 448, 1, 12, True)
```

