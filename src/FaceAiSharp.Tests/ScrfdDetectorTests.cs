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

    [Fact]
    public async Task ObamaFamily()
    {
        using var img = await Image.LoadAsync<Rgb24>("TestData/jpgs/obama_family.jpg");
        var res = _det.DetectFaces(img);
        res.Count.ShouldBe(4);

        // Rounding isn't the best choice here, as small differences could lead to another rounded value, e.g. 0.4 -> 0 and 0.5 -> 1.
        var ordered = res.OrderBy(x => x.Box.X).ThenBy(x => x.Box.Y)
            .Select(x => new
            {
                Confidence = x.Confidence.HasValue ? Math.Round(x.Confidence.Value, 1) : (double?)null,
                X = Math.Round(x.Box.X / 10), // we want to round to 10 pixels
                Y = Math.Round(x.Box.Y / 10),
                Width = Math.Round(x.Box.Width / 10),
                Height = Math.Round(x.Box.Height / 10),
                Landmarks = x.Landmarks?.Select(p => new PointF(MathF.Round(p.X / 10), MathF.Round(p.Y / 10)))?.ToList() ?? [],
            })
            .ToList();
        await Verify(ordered);
    }
}
