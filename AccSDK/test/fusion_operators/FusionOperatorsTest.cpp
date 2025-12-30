/*
* -------------------------------------------------------------------------
*  This file is part of the MultimodalSDK project.
* Copyright (c) 2025 Huawei Technologies Co.,Ltd.
*
* MultimodalSDK is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*
*           http://license.coscl.org.cn/MulanPSL2
*
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* -------------------------------------------------------------------------
 * Description: QwenFusion preprocess unit test with TEST_F
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "acc/fusion_operators/FusionOperators.h"
#include "acc/image/Image.h"
#include "acc/ErrorCode.h"
#include "acc/tensor/TensorDataType.h"
using namespace Acc;

namespace {

class QwenFusionTestFixture : public ::testing::Test {
protected:
    void SetUp() override
    {
        validImage = CreateValidImage();
        validConfig = MakeValidConfig();
    }

    std::shared_ptr<Image> CreateValidImage()
    {
        static std::vector<uint8_t> buffer(1920 * 1080 * 3, 100);
        return std::make_shared<Image>(std::shared_ptr<void>(&buffer[0], [](void*) {}), std::vector<size_t>{1920, 1080},
                                       ImageFormat::RGB, DataType::UINT8, "cpu");
    }

    QwenPreprocessConfig MakeValidConfig()
    {
        QwenPreprocessConfig cfg;
        cfg.mean = {0.5f, 0.5f, 0.5f};
        cfg.std = {1.0f, 1.0f, 1.0f};
        cfg.resizeW = 224;
        cfg.resizeH = 224;
        return cfg;
    }

    std::shared_ptr<Image> validImage;
    QwenPreprocessConfig validConfig;
    FusionOperator fusion;
};

// -------------------- Unit Tests --------------------

TEST_F(QwenFusionTestFixture, Empty_Images_Should_Fail)
{
    std::vector<std::shared_ptr<Image>> images;
    std::vector<Tensor> outputs;
    EXPECT_NE(fusion.Qwen2VLImagePreprocess(images, validConfig, outputs), SUCCESS);
}

TEST_F(QwenFusionTestFixture, Invalid_MeanStd_Should_Fail)
{
    std::vector<std::shared_ptr<Image>> images = {validImage};
    std::vector<Tensor> outputs;

    QwenPreprocessConfig cfg = validConfig;
    cfg.mean = {0.5f, 0.5f};
    EXPECT_NE(fusion.Qwen2VLImagePreprocess(images, cfg, outputs), SUCCESS);

    cfg = validConfig;
    cfg.std = {1.0f, 1.0f};
    EXPECT_NE(fusion.Qwen2VLImagePreprocess(images, cfg, outputs), SUCCESS);
}

TEST_F(QwenFusionTestFixture, Std_Values_Zero_Or_Negative_Should_Fail)
{
    std::vector<std::shared_ptr<Image>> images = {validImage};
    std::vector<Tensor> outputs;

    QwenPreprocessConfig cfg = validConfig;
    cfg.std[0] = 0.0f;
    EXPECT_NE(fusion.Qwen2VLImagePreprocess(images, cfg, outputs), SUCCESS);

    cfg = validConfig;
    cfg.std[0] = -1.0f;
    EXPECT_NE(fusion.Qwen2VLImagePreprocess(images, cfg, outputs), SUCCESS);
}

TEST_F(QwenFusionTestFixture, Resize_Dimensions_Invalid_Should_Fail)
{
    std::vector<std::shared_ptr<Image>> images = {validImage};
    std::vector<Tensor> outputs;

    QwenPreprocessConfig cfg = validConfig;
    cfg.resizeW = -1;
    EXPECT_NE(fusion.Qwen2VLImagePreprocess(images, cfg, outputs), SUCCESS);

    cfg = validConfig;
    cfg.resizeH = 0;
    EXPECT_NE(fusion.Qwen2VLImagePreprocess(images, cfg, outputs), SUCCESS);
}

TEST_F(QwenFusionTestFixture, Preprocess_Success)
{
    auto img2 = CreateValidImage();
    std::vector<std::shared_ptr<Image>> images = {validImage, img2};
    std::vector<Tensor> outputs;

    EXPECT_EQ(fusion.Qwen2VLImagePreprocess(images, validConfig, outputs), SUCCESS);
    EXPECT_EQ(outputs.size(), images.size());
}

TEST_F(QwenFusionTestFixture, Preprocess_Different_Sizes_And_Layouts)
{
    std::vector<std::shared_ptr<Image>> images;
    std::vector<Tensor> outputs;
    images.push_back(validImage);
    static std::vector<uint8_t> buffer(800 * 600 * 3, 50);
    auto img2 = std::make_shared<Image>(std::shared_ptr<void>(&buffer[0], [](void*) {}), std::vector<size_t>{800, 600},
                                        ImageFormat::RGB, DataType::UINT8, "cpu");
    images.push_back(img2);

    QwenPreprocessConfig cfg = validConfig;

    EXPECT_EQ(fusion.Qwen2VLImagePreprocess(images, cfg, outputs), SUCCESS);
    EXPECT_EQ(outputs.size(), images.size());
}

} // namespace

// -------------------- main --------------------
int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
