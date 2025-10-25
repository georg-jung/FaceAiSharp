// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using NumSharp;
using SixLabors.ImageSharp;

namespace FaceAiSharp.Tests;

public class ScrfdTensorOps
{
    [Theory]
    [InlineData(320, 240, 8, 2)]
    [InlineData(640, 480, 16, 2)]
    public void Test(int inputWidth, int inputHeight, int stride, int numAnchors)
    {
        var numSharpRes = GenerateAnchorCentersNumSharp(inputWidth, inputHeight, stride, numAnchors);
        var numSharpArray = numSharpRes.Data<float>().ToArray();
        var anchorCenterCnt = (inputWidth / stride) * (inputHeight / stride) * numAnchors;
        Assert.Equal(anchorCenterCnt * 2, numSharpArray.Length);

        var ours = new float[anchorCenterCnt * 2];
        for (var anchorIdx = 0; anchorIdx < anchorCenterCnt; anchorIdx++)
        {
            var (x, y) = ScrfdDetector.GetAnchorCenter(new Size(inputWidth, inputHeight), stride, numAnchors, anchorIdx);
            ours[(anchorIdx * 2) + 0] = x;
            ours[(anchorIdx * 2) + 1] = y;
        }

        Assert.Equal(numSharpArray, ours);
    }

    private static NDArray GenerateAnchorCentersNumSharp(int inputWidth, int inputHeight, int stride, int numAnchors)
    {
        // translated from https://github.com/deepinsight/insightface/blob/f091989568cad5a0244e05be1b8d58723de210b0/detection/scrfd/tools/scrfd.py#L185
        var height = inputHeight / stride;
        var width = inputWidth / stride;
        var (mgrid1, mgrid2) = np.mgrid(np.arange(height), np.arange(width));
        var anchorCenters = np.stack([mgrid2, mgrid1], axis: -1).astype(np.float32);
        anchorCenters = (anchorCenters * stride).reshape(-1, 2);
        if (numAnchors > 1)
        {
            anchorCenters = np.stack([anchorCenters, anchorCenters], axis: 1).reshape(-1, 2);
        }

        return anchorCenters;
    }
}
