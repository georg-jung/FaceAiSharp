// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Attributes;
using FaceAiSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Benchmarks.Regression;

[BenchmarkCategory("regression")]
public class EyeState
{
    private OpenVinoOpenClosedEye0001 _eyeStateDetector = null!;
    private Image<Rgb24> _eye32 = null!;

    [GlobalSetup]
    public void Setup()
    {
        _eyeStateDetector = new OpenVinoOpenClosedEye0001(new OpenVinoOpenClosedEye0001Options
        {
            ModelPath = BenchmarkData.ModelPath("open_closed_eye.onnx"),
        });

        using var portrait = BenchmarkData.LoadImage("Barack_Obama_03.jpg");
        _eye32 = portrait.Clone(op => op.Resize(32, 32));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _eyeStateDetector.Dispose();
        _eye32.Dispose();
    }

    [Benchmark]
    public bool IsOpen() => _eyeStateDetector.IsOpen(_eye32);
}
