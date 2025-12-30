# AccData
## 简介
AccData定位为AI数据处理套件，主要面向AI训练推理等场景，提供高效易用的数据处理加速服务。AccData将数据处理流程抽象为pipeline，并采用操作融合、数据预取等优化手段，达成倍数级优化数据处理性能的目标。

## 构建
```
git clone --recursive https://szv-y.codehub.huawei.com/BeiMing/service_domain/acceleration_library/a_k/acc_data.git
bash build.sh
```
### 拉取submodule命令：
通过git命令行 ```git submodule update --init --recursive --remote```拉代码

## 安装
```
pip3 install dist/accdata-*.whl
```

## 单元测试
### 编译执行UT
+ 编译运行冒烟测试（py）：`bash buildscript/run_dt.sh -s`
+ 编译运行全量测试（py）：`bash buildscript/run_dt.sh -a`
+ 编译运行冒烟测试（cpp）：`bash buildscript/run_dt.sh -s -t cpp`
+ 编译运行全量测试（cpp）：`bash buildscript/run_dt.sh -a -t cpp`
+ 编译运行冒烟测试（cpp + py）：`bash buildscript/run_dt.sh -s -t all`
+ 编译运行全量测试（cpp + py）：`bash buildscript/run_dt.sh -a -t all`
+ 无需编译（以cpp为例）：`bash buildscript/run_dt.sh -b false -t cpp`


### 执行dt-fuzz
+ `cd test/cpp`
+ `bash run_fuzz_test.sh`