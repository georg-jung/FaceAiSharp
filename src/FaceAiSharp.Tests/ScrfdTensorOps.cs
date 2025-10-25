// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using NumSharp;
using SixLabors.ImageSharp;
using Xunit.Sdk;

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
        var x = numSharpRes.Shape;
        var ours = ScrfdDetector.GenerateAnchorCenters(new Size(inputWidth, inputHeight), stride, numAnchors);
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
