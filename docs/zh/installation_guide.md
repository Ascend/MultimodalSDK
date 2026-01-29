# 安装部署<a name="ZH-CN_TOPIC_0000002423437800"></a>

## 获取安装包<a name="ZH-CN_TOPIC_0000002456876697"></a>

请参考本章获取所需软件包和对应的数字签名文件，下载本软件即表示您同意[华为企业业务最终用户许可协议（EULA）](https://e.huawei.com/cn/about/eula)的条款和条件。

**表 1**  软件包

|组件名称|软件包名称|
|--|--|
|Multimodal SDK|多模态软件包|


**软件数字签名验证<a name="section10830205518487"></a>**

为了防止软件包在传递过程中或存储期间被恶意篡改，下载软件包时请下载对应的数字签名文件用于完整性验证。

在软件包下载之后，请参考《OpenPGP签名验证指南》，对下载的软件包进行PGP数字签名校验。如果校验失败，请勿使用该软件包并联系华为技术支持工程师解决。

使用软件包安装/升级前，也需要按照上述过程，验证软件包的数字签名，确保软件包未被篡改。

运营商客户请访问：[https://support.huawei.com/carrier/digitalSignatureAction](https://support.huawei.com/carrier/digitalSignatureAction)

企业客户请访问：[https://support.huawei.com/enterprise/zh/tool/software-digital-signature-openpgp-validation-tool-TL1000000054](https://support.huawei.com/enterprise/zh/tool/software-digital-signature-openpgp-validation-tool-TL1000000054)


## 安装依赖<a name="ZH-CN_TOPIC_0000002423192164"></a>

### 安装Ubuntu系统依赖<a name="ZH-CN_TOPIC_0000002423192152"></a>

Ubuntu系统环境中所需依赖名称、对应版本及获取建议请参见[表1](#table20540329125613)。

**表 1** Ubuntu系统依赖名称对应版本

<a name="table20540329125613"></a>
|依赖名称|版本建议|获取建议|
|--|--|--|
|Python|3.9及以上|建议通过获取源码包编译安装。|
|CMake|3.14及以上|建议通过包管理模块安装。|
|Make|4.1及以上|建议通过包管理模块安装。|
|GCC|9.4及以上|建议通过包管理模块安装。|


参考如下命令，检查是否已安装GCC、Make、CMake等依赖软件。

```
gcc --version
make --version
cmake --version
python3 --version
```

若分别返回如下信息，说明相应软件已安装（以下回显仅为示例，请以实际情况为准）。

```
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
GNU Make 4.1
cmake version 3.14.2
Python 3.10.12
```


### 安装NPU驱动固件和CANN<a name="ZH-CN_TOPIC_0000002456910773"></a>

**下载依赖软件包<a name="section119752030133014"></a>**

下载本软件即表示您同意[华为企业业务最终用户许可协议（EULA）](https://e.huawei.com/cn/about/eula)的条款和条件。

**表 1**  软件包清单

<a name="table1624445102817"></a>
<table><thead align="left"><tr id="row724174542813"><th class="cellrowborder" valign="top" width="21.72%" id="mcps1.2.4.1.1"><p id="p112413454282"><a name="p112413454282"></a><a name="p112413454282"></a>软件类型</p>
</th>
<th class="cellrowborder" valign="top" width="53.68000000000001%" id="mcps1.2.4.1.2"><p id="p725124572814"><a name="p725124572814"></a><a name="p725124572814"></a>软件包名称</p>
</th>
<th class="cellrowborder" valign="top" width="24.6%" id="mcps1.2.4.1.3"><p id="p919101122613"><a name="p919101122613"></a><a name="p919101122613"></a>获取方式</p>
</th>
</tr>
</thead>
<tbody><tr id="row20251045102819"><td class="cellrowborder" valign="top" width="21.72%" headers="mcps1.2.4.1.1 "><p id="p73893235281"><a name="p73893235281"></a><a name="p73893235281"></a>昇腾NPU驱动</p>
</td>
<td class="cellrowborder" valign="top" width="53.68000000000001%" headers="mcps1.2.4.1.2 "><p id="p2444175718384"><a name="p2444175718384"></a><a name="p2444175718384"></a>Ascend-hdk-<em id="i151577715547"><a name="i151577715547"></a><a name="i151577715547"></a>{npu_type}</em>-npu-driver_<em id="i171131251111111"><a name="i171131251111111"></a><a name="i171131251111111"></a>{version}</em>_linux-<em id="i19190436126"><a name="i19190436126"></a><a name="i19190436126"></a>{arch}</em>.run</p>
</td>
<td class="cellrowborder" rowspan="3" valign="top" width="24.6%" headers="mcps1.2.4.1.3 "><p id="p1130893682616"><a name="p1130893682616"></a><a name="p1130893682616"></a>单击<a href="https://www.hiascend.com/developer/download/commercial/result?module=cann" target="_blank" rel="noopener noreferrer">获取链接</a>，在左侧配套资源的“编辑资源选择”中进行配置，筛选配套的软件包，确认版本信息后获取所需软件包。</p>
</td>
</tr>
<tr id="row225114514289"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p339015237286"><a name="p339015237286"></a><a name="p339015237286"></a>昇腾NPU固件</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p14444165713389"><a name="p14444165713389"></a><a name="p14444165713389"></a>Ascend-hdk-<em id="i1016132425913"><a name="i1016132425913"></a><a name="i1016132425913"></a>{npu_type}</em>-npu-firmware_<em id="i1269851591216"><a name="i1269851591216"></a><a name="i1269851591216"></a>{version}</em>.run</p>
</td>
</tr>
<tr id="row1838172311404"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p81248297404"><a name="p81248297404"></a><a name="p81248297404"></a>CANN软件包</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p91244293404"><a name="p91244293404"></a><a name="p91244293404"></a>Ascend-cann-toolkit_<em id="i68531246101211"><a name="i68531246101211"></a><a name="i68531246101211"></a>{version}</em>_linux-<em id="i12975052151210"><a name="i12975052151210"></a><a name="i12975052151210"></a>{arch}</em>.run</p>
</td>
</tr>
</tbody>
</table>


>[!NOTE] 说明 
>-   \{npu\_type\}表示芯片名称。
>-   \{version\}表示软件版本号。
>-   \{arch\}表示CPU架构。

**安装NPU驱动固件和CANN<a name="section2121626113418"></a>**

1.  参考《CANN 软件安装指南》中的“安装NPU驱动和固件”章节（商用版）或“安装NPU驱动和固件”章节（社区版）安装NPU驱动固件。
2.  参考《CANN 软件安装指南》的“安装CANN”章节（商用版）或《CANN 软件安装指南》的“安装CANN”章节（社区版）安装CANN。

    >[!NOTE] 说明
    >-   安装CANN（Toolkit），NPU驱动固件和安装Multimodal SDK的用户需为同一用户，建议为普通用户。
    >-   安装CANN时，为确保Multimodal SDK正常使用，CANN的相关依赖也需要一并安装。


### 安装Python软件包依赖<a name="ZH-CN_TOPIC_0000002466622269"></a>

使用Multimodal SDK相关功能还需安装[表1](#table20540329125613)中的所有依赖。若使用[patcher](./api/patcher.md)中的patcher环节，请安装[patcher](./api/patcher.md)中的镜像，无需额外安装[表1](#table20540329125613)中的依赖。

**表 1**  依赖名称对应版本

<a name="table20540329125613"></a>
|依赖名称|版本建议|获取建议|
|--|--|--|
|transformers|4.51.3|建议通过pip获取。|
|pillow|11.2.1|建议通过pip获取。|
|numpy|1.26.4|建议通过pip获取。|
|torch|2.5.1|建议通过pip获取。|


**注意事项<a name="section1297475493911"></a>**

如需安装Multimodal SDK软件包以外的第三方软件，请注意及时升级最新版本，关注并修补存在的漏洞。



## 安装Multimodal SDK<a name="ZH-CN_TOPIC_0000002456790913"></a>

**安装须知<a name="section3134195618512"></a>**

-   安装和运行Multimodal SDK的用户，需要满足：
    -   安装和运行Multimodal SDK的用户建议为普通用户。
    -   安装和运行Multimodal SDK的用户需为同一用户。
    -   安装CANN（Toolkit），NPU驱动固件和安装Multimodal SDK的用户需为同一用户，建议为普通用户。

-   软件包的安装、升级、卸载及版本查询相关的日志会保存至“\~/log/mindxsdk/deployment.log“文件；完整性校验、提取文件、tar命令访问相关的日志会保存至“\~/log/makeself/makeself.log“文件。用户可查看相应文件，完成后续的日志跟踪及审计。

**安装准备<a name="section15633111461011"></a>**

确保安装环境中已执行CANN环境变量配置脚本，使环境变量生效。具体执行路径，请按照实际安装为准。

```
# 安装toolkit包
. /usr/local/Ascend/cann/set_env.sh #此处为CANN默认安装路径，根据实际安装路径修改
```

**安装步骤<a name="section12327567584"></a>**

1.  以软件包的安装用户登录安装环境。
2.  将Multimodal SDK软件包上传到安装环境的任意路径下并进入软件包所在路径。
3.  增加对软件包的可执行权限。

    ```
    chmod u+x Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run
    ```

4.  执行如下命令，校验软件包的一致性和完整性。

    ```
    ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --check
    ```

    如果系统没有shasum或者sha256sum工具则会校验失败，此时需要自行安装shasum或者sha256sum工具。

    若显示如下信息，说明软件包已通过校验。

    ```
    Verifying archive integrity...  100%   SHA256 checksums are OK. All good.    
    ```

5.  创建Multimodal SDK软件包的安装路径。不建议在“/tmp“路径下安装。
    -   若用户未指定安装路径，软件会默认安装到Multimodal SDK软件包所在的路径。
    -   若用户想指定安装路径，需要先创建安装路径。以安装路径“/home/work/Mind\_SDK“为例：

        ```
        mkdir -p /home/work/Mind_SDK
        ```

6.  进入Multimodal SDK软件包所在路径，参考以下命令安装Multimodal SDK（安装路径的相关约束请参见[表1](#table1361972315353)）。安装Multimodal SDK时会弹出确认是否接受EULA协议的提示，若需要在安装时直接跳过该步骤，可在安装命令前增加**echo y |**，表示同意[华为企业业务最终用户许可协议（EULA）](https://e.huawei.com/cn/about/eula)**。**

    -   若用户指定了安装路径，将安装在指定的路径下。以安装路径“/home/work/Mind\_SDK“为例：

        ```
        ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --install --install-path=/home/work/Mind_SDK
        ```

        或者

        ```
        echo y | ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --install --install-path=/home/work/Mind_SDK
        ```

    -   若用户未指定安装路径，将安装在当前路径下。

        ```
        ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --install
        ```

        或者

        ```
        echo y | ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --install
        ```

    >[!NOTE] 说明 
    >--install安装命令同时支持输入可选参数，如[表1](#table1361972315353)所示。

7.  安装过程中提示“Do you accept the EULA to install Multimodal SDK?”时，输入Y或y，表示同意EULA协议，继续进行安装；输入其他字符时停止安装，退出程序。

    安装完成后，若显示如下信息，表示软件安装成功。

    ```
    Successfully installed Multimodal SDK
    ```

8.  进入Multimodal SDK安装路径下的“script“目录，执行以下命令使Multimodal SDK的环境变量生效。

    ```
    source set_env.sh
    ```

**相关参考<a name="section111812571483"></a>**

**表 1**  --install安装命令可选参数表

<a name="table1361972315353"></a>
<table><thead align="left"><tr id="row1719215818116"><th class="cellrowborder" valign="top" width="26.06%" id="mcps1.2.3.1.1"><p id="p2625159111215"><a name="p2625159111215"></a><a name="p2625159111215"></a>输入参数</p>
</th>
<th class="cellrowborder" valign="top" width="73.94%" id="mcps1.2.3.1.2"><p id="p18625792122"><a name="p18625792122"></a><a name="p18625792122"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001506619361_row1171151615219"><td class="cellrowborder" valign="top" width="26.06%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001506619361_p1671151616217"><a name="zh-cn_topic_0000001506619361_p1671151616217"></a><a name="zh-cn_topic_0000001506619361_p1671151616217"></a>--help | -h</p>
</td>
<td class="cellrowborder" valign="top" width="73.94%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001506619361_p12711416112113"><a name="zh-cn_topic_0000001506619361_p12711416112113"></a><a name="zh-cn_topic_0000001506619361_p12711416112113"></a>查询帮助信息。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001506619361_row4717168211"><td class="cellrowborder" valign="top" width="26.06%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001506619361_p1771171662116"><a name="zh-cn_topic_0000001506619361_p1771171662116"></a><a name="zh-cn_topic_0000001506619361_p1771171662116"></a>--info</p>
</td>
<td class="cellrowborder" valign="top" width="73.94%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001506619361_p127115169219"><a name="zh-cn_topic_0000001506619361_p127115169219"></a><a name="zh-cn_topic_0000001506619361_p127115169219"></a>查询包构建信息。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001506619361_row1711716172115"><td class="cellrowborder" valign="top" width="26.06%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001506619361_p1571616102112"><a name="zh-cn_topic_0000001506619361_p1571616102112"></a><a name="zh-cn_topic_0000001506619361_p1571616102112"></a>--list</p>
</td>
<td class="cellrowborder" valign="top" width="73.94%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001506619361_p2071101611219"><a name="zh-cn_topic_0000001506619361_p2071101611219"></a><a name="zh-cn_topic_0000001506619361_p2071101611219"></a>查询文件列表。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001506619361_row20711216152120"><td class="cellrowborder" valign="top" width="26.06%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001506619361_p9711916142116"><a name="zh-cn_topic_0000001506619361_p9711916142116"></a><a name="zh-cn_topic_0000001506619361_p9711916142116"></a>--check</p>
</td>
<td class="cellrowborder" valign="top" width="73.94%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001506619361_p1671161612111"><a name="zh-cn_topic_0000001506619361_p1671161612111"></a><a name="zh-cn_topic_0000001506619361_p1671161612111"></a>查询包完整性。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001506619361_row13711316142116"><td class="cellrowborder" valign="top" width="26.06%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001506619361_p971181614210"><a name="zh-cn_topic_0000001506619361_p971181614210"></a><a name="zh-cn_topic_0000001506619361_p971181614210"></a>--quiet | -q</p>
</td>
<td class="cellrowborder" valign="top" width="73.94%" headers="mcps1.2.3.1.2 "><p id="p9289230777"><a name="p9289230777"></a><a name="p9289230777"></a>启用静默模式，并默认同意<a href="https://e.huawei.com/cn/about/eula" target="_blank" rel="noopener noreferrer">华为企业业务最终用户许可协议（EULA）</a>。</p>
<p id="zh-cn_topic_0000001506619361_p271101632111"><a name="zh-cn_topic_0000001506619361_p271101632111"></a><a name="zh-cn_topic_0000001506619361_p271101632111"></a>需要和<span class="parmname" id="parmname7774144681516"><a name="parmname7774144681516"></a><a name="parmname7774144681516"></a>“--install”</span>或<span class="parmname" id="parmname16806125014150"><a name="parmname16806125014150"></a><a name="parmname16806125014150"></a>“--upgrade”</span>参数配合使用。</p>
</td>
</tr>
<tr id="row14840144218392"><td class="cellrowborder" valign="top" width="26.06%" headers="mcps1.2.3.1.1 "><p id="p9841242163912"><a name="p9841242163912"></a><a name="p9841242163912"></a>--nox11</p>
</td>
<td class="cellrowborder" valign="top" width="73.94%" headers="mcps1.2.3.1.2 "><p id="p1284174213915"><a name="p1284174213915"></a><a name="p1284174213915"></a>废弃接口，无实际作用</p>
</td>
</tr>
<tr id="row12833191344013"><td class="cellrowborder" valign="top" width="26.06%" headers="mcps1.2.3.1.1 "><p id="p983461334016"><a name="p983461334016"></a><a name="p983461334016"></a>--noexec</p>
</td>
<td class="cellrowborder" valign="top" width="73.94%" headers="mcps1.2.3.1.2 "><p id="p38341413134017"><a name="p38341413134017"></a><a name="p38341413134017"></a>不运行嵌入的脚本</p>
</td>
</tr>
<tr id="row20502252174018"><td class="cellrowborder" valign="top" width="26.06%" headers="mcps1.2.3.1.1 "><p id="p4503145214014"><a name="p4503145214014"></a><a name="p4503145214014"></a>--extract=&lt;path&gt;</p>
</td>
<td class="cellrowborder" valign="top" width="73.94%" headers="mcps1.2.3.1.2 "><p id="p58181517124117"><a name="p58181517124117"></a><a name="p58181517124117"></a>直接提取到目标目录（绝对路径或相对路径）</p>
<p id="p14818101712419"><a name="p14818101712419"></a><a name="p14818101712419"></a>通常与 --noexec 选项一起使用，仅用于提取文件而不运行它们</p>
</td>
</tr>
<tr id="row20213115320418"><td class="cellrowborder" valign="top" width="26.06%" headers="mcps1.2.3.1.1 "><p id="p19213125384114"><a name="p19213125384114"></a><a name="p19213125384114"></a>--tar arg1 [arg2 ...]</p>
</td>
<td class="cellrowborder" valign="top" width="73.94%" headers="mcps1.2.3.1.2 "><p id="p17521171464212"><a name="p17521171464212"></a><a name="p17521171464212"></a>通过 tar 命令访问归档文件的内容</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001506619361_row2071121682119"><td class="cellrowborder" valign="top" width="26.06%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001506619361_p0711316182110"><a name="zh-cn_topic_0000001506619361_p0711316182110"></a><a name="zh-cn_topic_0000001506619361_p0711316182110"></a>--install</p>
</td>
<td class="cellrowborder" valign="top" width="73.94%" headers="mcps1.2.3.1.2 "><p id="p67991756045"><a name="p67991756045"></a><a name="p67991756045"></a>Multimodal SDK软件包安装操作命令。</p>
<a name="ul498815593414"></a><a name="ul498815593414"></a><ul id="ul498815593414"><li>当前路径和安装路径不能存在非法字符，仅支持大小写字母、数字、-_./特殊字符。</li><li>安装路径下不能存在名为multimodal的文件或文件夹。</li><li>若存在名为multimodal的软链接，则会被覆盖。</li></ul>
</td>
</tr>
<tr id="zh-cn_topic_0000001506619361_row207120161217"><td class="cellrowborder" valign="top" width="26.06%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001506619361_p187191642111"><a name="zh-cn_topic_0000001506619361_p187191642111"></a><a name="zh-cn_topic_0000001506619361_p187191642111"></a>--install-path=<em id="zh-cn_topic_0000001506619361_i11651455321"><a name="zh-cn_topic_0000001506619361_i11651455321"></a><a name="zh-cn_topic_0000001506619361_i11651455321"></a>&lt;path&gt;</em></p>
</td>
<td class="cellrowborder" valign="top" width="73.94%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001506619361_p20733533278"><a name="zh-cn_topic_0000001506619361_p20733533278"></a><a name="zh-cn_topic_0000001506619361_p20733533278"></a>（可选）自定义软件包安装根目录。如未设置，默认为当前命令执行所在目录。</p>
<a name="zh-cn_topic_0000001506619361_ul810411616313"></a><a name="zh-cn_topic_0000001506619361_ul810411616313"></a><ul id="zh-cn_topic_0000001506619361_ul810411616313"><li>建议用户使用绝对路径安装Multimodal SDK，指定安装路径时请避免使用相对路径。</li><li>需要和<span class="parmname" id="parmname06541154101515"><a name="parmname06541154101515"></a><a name="parmname06541154101515"></a>“--install”</span>或<span class="parmname" id="parmname711115589158"><a name="parmname711115589158"></a><a name="parmname711115589158"></a>“--upgrade”</span>参数配合使用。</li><li>与<span class="parmname" id="parmname556791051613"><a name="parmname556791051613"></a><a name="parmname556791051613"></a>“--upgrade”</span>参数配合使用时，--install-path代表旧软件包的安装目录，并在该目录下执行升级。</li><li>传入的路径参数不能存在非法字符，仅支持大小写字母、数字、-_./特殊字符。</li></ul>
</td>
</tr>
<tr id="zh-cn_topic_0000001506619361_row67181614212"><td class="cellrowborder" valign="top" width="26.06%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001506619361_p1571131622115"><a name="zh-cn_topic_0000001506619361_p1571131622115"></a><a name="zh-cn_topic_0000001506619361_p1571131622115"></a>--upgrade</p>
</td>
<td class="cellrowborder" valign="top" width="73.94%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001506619361_p772131652119"><a name="zh-cn_topic_0000001506619361_p772131652119"></a><a name="zh-cn_topic_0000001506619361_p772131652119"></a>Multimodal SDK软件包升级操作命令。升级需要确保已经安装过目录完整的Multimodal SDK。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001506619361_row1172816182118"><td class="cellrowborder" valign="top" width="26.06%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001506619361_p17281611210"><a name="zh-cn_topic_0000001506619361_p17281611210"></a><a name="zh-cn_topic_0000001506619361_p17281611210"></a>--version</p>
</td>
<td class="cellrowborder" valign="top" width="73.94%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001506619361_p17241642114"><a name="zh-cn_topic_0000001506619361_p17241642114"></a><a name="zh-cn_topic_0000001506619361_p17241642114"></a>查询软件包Multimodal SDK版本。</p>
</td>
</tr>
</tbody>
</table>


>[!NOTE] 说明 
>以下参数未展示在--help参数中，用户请勿直接使用。
>-   --xwin：使用xwin模式运行。
>-   --phase2：要求执行第二步动作。

# 卸载<a name="ZH-CN_TOPIC_0000002423352016"></a>

**操作步骤<a name="section12002371094"></a>**

1.  进入Multimodal SDK的安装路径，确认Multimodal SDK目录下“script”目录中的“uninstall.sh”脚本是否有可执行权限。

    ```
    cd multimodal/script
    ls -l uninstall.sh
    ```

    若脚本没有可执行权限，请执行如下命令，赋予“uninstall.sh”脚本可执行权限。

    ```
    chmod u+x uninstall.sh
    ```

2.  执行如下命令，开始执行卸载。在执行卸载脚本时，脚本会卸载已安装的python whl包并删除安装目录。

    ```
    ./uninstall.sh
    ```

    >[!NOTE] 说明 
    >使用“uninstall.sh”脚本进行卸载操作仅适用于正常安装途径，且安装后未对安装文件结构进行修改，如需解决安装异常等情况，请通过完全删除安装目录下任何有关multimodal的文件夹，以及pip uninstall mm卸载安装的Python包文件。


