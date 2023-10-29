// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Shouldly;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceAiSharp.Tests;

public class ScrfdDetectorTests : IClassFixture<ScrfdDetectorFixture>
{
    private readonly ScrfdDetector _det;

    public ScrfdDetectorTests(ScrfdDetectorFixture detectorFixture)
    {
        _det = detectorFixture.Detector;
    }

    [Fact]
    public void BlankImageNoFaces()
    {
        using var blankImg = new Image<Rgb24>(640, 640, new Rgb24(255, 255, 255));
        _det.DetectFaces(blankImg).ShouldBeEmpty();
    }
}
