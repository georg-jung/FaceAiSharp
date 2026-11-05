// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Text.Json;
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
    public void BlankImageNoLandmarks()
    {
        using var blankImg = new Image<Rgb24>(640, 640, new Rgb24(255, 255, 255));
        IFaceLandmarksDetector det = _det;
        det.DetectLandmarks(blankImg).ShouldBeEmpty();
    }

    [Fact]
    public async Task LandmarksOfMostConfidentFace()
    {
        using var img = await Image.LoadAsync<Rgb24>("TestData/jpgs/obama_family.jpg");
        IFaceLandmarksDetector det = _det;
        var expected = _det.DetectFaces(img).MaxBy(x => x.Confidence).Landmarks;

        var landmarks = det.DetectLandmarks(img);

        landmarks.Count.ShouldBe(5);
        landmarks.ShouldBe(expected);
        det.GetLeftEyeCenter(landmarks).ShouldBe(landmarks[0]);
        det.GetRightEyeCenter(landmarks).ShouldBe(landmarks[1]);
    }

    [Fact]
    public void EyeCentersOfEmptyLandmarksThrow()
    {
        IFaceLandmarksDetector det = _det;

        // Should.Throw<ArgumentException> would also accept the indexer's ArgumentOutOfRangeException, so check the exact type.
        var left = Should.Throw<ArgumentException>(() => det.GetLeftEyeCenter([]));
        left.ShouldBeOfType<ArgumentException>();
        left.ParamName.ShouldBe("landmarks");

        var right = Should.Throw<ArgumentException>(() => det.GetRightEyeCenter([]));
        right.ShouldBeOfType<ArgumentException>();
        right.ParamName.ShouldBe("landmarks");
    }

    [Fact]
    public void MisspelledNonMaxSuppressionAliasForwards()
    {
#pragma warning disable CS0618 // Type or member is obsolete
        var opts = new ScrfdDetectorOptions { NonMaxSupressionThreshold = 0.25f };
        opts.NonMaxSuppressionThreshold.ShouldBe(0.25f);
        opts.NonMaxSuppressionThreshold = 0.3f;
        opts.NonMaxSupressionThreshold.ShouldBe(0.3f);
#pragma warning restore CS0618 // Type or member is obsolete

        var json = JsonSerializer.Serialize(opts);
        json.ShouldContain("\"NonMaxSuppressionThreshold\"");
        json.ShouldNotContain("NonMaxSupressionThreshold");
    }

    [Fact]
    public void MisspelledNonMaxSuppressionNameIsReadFromLegacyJson()
    {
        var legacy = JsonSerializer.Deserialize<ScrfdDetectorOptions>("""{"NonMaxSupressionThreshold":0.1}""")!;
        legacy.NonMaxSuppressionThreshold.ShouldBe(0.1f);

        var current = JsonSerializer.Deserialize<ScrfdDetectorOptions>("""{"NonMaxSuppressionThreshold":0.2}""")!;
        current.NonMaxSuppressionThreshold.ShouldBe(0.2f);

        var roundTripped = JsonSerializer.Deserialize<ScrfdDetectorOptions>(JsonSerializer.Serialize(current))!;
        roundTripped.NonMaxSuppressionThreshold.ShouldBe(0.2f);
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
