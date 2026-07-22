// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Attributes;
using FaceAiSharp;
using NumSharp;
using SixLabors.ImageSharp;

namespace Benchmarks.Regression;

[BenchmarkCategory("regression")]
public class NonMaxSuppression
{
    private IReadOnlyList<FaceDetectorResult> _crowd = null!;
    private IReadOnlyList<FaceDetectorResult> _group = null!;
    private IReadOnlyList<FaceDetectorResult> _portrait = null!;

    [GlobalSetup]
    public void Setup()
    {
        _crowd = LoadDetections("crowd.npy");
        _group = LoadDetections("group.npy");
        _portrait = LoadDetections("portrait.npy");
    }

    [Benchmark]
    public List<int> Crowd() => ScrfdDetector.NonMaxSuppression(_crowd, 0.4f);

    [Benchmark]
    public List<int> Group() => ScrfdDetector.NonMaxSuppression(_group, 0.4f);

    [Benchmark]
    public List<int> Portrait() => ScrfdDetector.NonMaxSuppression(_portrait, 0.4f);

    private static IReadOnlyList<FaceDetectorResult> LoadDetections(string npyFileName)
    {
        var dets = np.load(BenchmarkData.TestDataPath("NMS", npyFileName));
        var numDetections = dets.shape[0];
        var results = new List<FaceDetectorResult>(numDetections);
        var x1s = dets[":, 0"].ToArray<float>();
        var y1s = dets[":, 1"].ToArray<float>();
        var x2s = dets[":, 2"].ToArray<float>();
        var y2s = dets[":, 3"].ToArray<float>();
        var scores = dets[":, 4"].ToArray<float>();
        for (var i = 0; i < numDetections; i++)
        {
            var box = new RectangleF(
                x: x1s[i],
                y: y1s[i],
                width: x2s[i] - x1s[i],
                height: y2s[i] - y1s[i]);
            results.Add(new FaceDetectorResult(box, null, scores[i]));
        }

        return results;
    }
}
