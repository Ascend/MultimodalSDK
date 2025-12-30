# ACC_SDK

## 代码目录

```

.
├── include                           # 对外公开暴露头文件
├── python                            # python代码
├── README.md
├── sample                            # 样例代码
├── doc                               # 设计文档
├── source                            # 源码
│   ├── audio                         # 音频模态
│   ├── core                          # 核心组件
│   │   ├── common                    # 公共模块，例如错误码
│   │   ├── framework                 # 异构算子调用模块
│   │   ├── resource                  # 全局资源管理模块
│   │   └── Tensor.cpp                # Tensor数据类
│   ├── image                         # 图像模态
│   │   ├── Image.cpp                 # 数据类
│   │   ├── ImageOps.cpp              # 加速接口
│   │   └── ops                       # 自定义算子
│   ├── inc                           # 内部头文件
    ├── py                            # swig绑定c++接口
│   ├── tensor
│   │   ├── ops                       # Tensor自定义算子
│   │   └── TensorOps.cpp             # Tensor加速接口
│   ├── utils                         # 工具类
│   └── video                         # 视频模态
└── test                              # 测试代码

```