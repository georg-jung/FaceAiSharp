// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Codeuctivity.ImageSharpCompare;
using Shouldly;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceAiSharp.Tests;

public class ApplicationsTests(ScrfdDetectorFixture scrfdFixture, OpenVinoOpenClosedEye0001Fixture openClosedFixture)
    : IClassFixture<ScrfdDetectorFixture>,
        IClassFixture<OpenVinoOpenClosedEye0001Fixture>
{
    [Fact]
    public async Task BlurFaces()
    {
        using var img = await Image.LoadAsync<Rgb24>("TestData/jpgs/Barack_Obama_03.jpg");
        using var clone = img.Clone();
        scrfdFixture.Detector.BlurFaces(img);
        ImageSharpCompare.CalcDiff(img, clone).PixelErrorPercentage.ShouldBeGreaterThan(0.05);
    }

    [Fact]
    public async Task CropProfilePicture()
    {
        using var img = await Image.LoadAsync<Rgb24>("TestData/jpgs/Barack_Obama_03.jpg");
        using var clone = img.Clone();
        scrfdFixture.Detector.CropProfilePicture(img);
        ImageSharpCompare.CalcDiff(img, clone, ResizeOption.Resize).PixelErrorPercentage.ShouldBeGreaterThan(0.2);
    }

    [Fact]
    public async Task CountFaces()
    {
        using var img = await Image.LoadAsync<Rgb24>("TestData/jpgs/Barack_Obama_03.jpg");
        scrfdFixture.Detector.CountFaces(img).ShouldBe(1);
    }

    [Fact]
    public async Task CountEyeStates()
    {
        using var img = await Image.LoadAsync<Rgb24>("TestData/jpgs/Barack_Obama_03.jpg");
        scrfdFixture.Detector.CountEyeStates(openClosedFixture.OpenClosed, img)
            .ShouldBe((1, 2, 0));
    }
}
