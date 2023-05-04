// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using BenchmarkDotNet.Attributes;
using FaceAiSharp;
using FaceAiSharp.Abstractions;
using FaceAiSharp.Extensions;
using FaceONNX;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Benchmarks;

[MemoryDiagnoser]
public class Scrfd
{
    private readonly Image _img = Image.Load(@"C:\Users\georg\facePics\avGroup.jpg");
    private readonly Image<Rgb24> _preprocImg;
    private readonly ScrfdDetector _scrfd1;
    private readonly ScrfdDetector _scrfd2;
    private readonly DenseTensor<float> _imgTensor;

    public Scrfd()
    {
        var x = _img.EnsureProperlySized<Rgb24>(
            new ResizeOptions()
            {
                Size = new Size(640),
                Position = AnchorPositionMode.TopLeft,
                Mode = ResizeMode.BoxPad,
                PadColor = Color.Black,
            },
            false);

        _preprocImg = x.Image;
        var opts = new MemoryCacheOptions();
        var iopts = Options.Create(opts);
        var c1 = new MemoryCache(iopts);
        var c2 = new MemoryCache(iopts);
        _imgTensor = ScrfdDetector.CreateImageTensor(_preprocImg);
        _scrfd1 = new(
            c1,
            new()
            {
                ModelPath = @"C:\Users\georg\OneDrive\Dokumente\ScrfdOnnx\scrfd_2.5g_bnkps_shape640x640.onnx",
                AutoResizeInputToModelDimensions = false,
            });

        _scrfd2 = new(
            c2,
            new()
            {
                ModelPath = @"C:\Users\georg\OneDrive\Dokumente\ScrfdOnnx\scrfd_2.5g_bnkps_dyn.onnx",
                AutoResizeInputToModelDimensions = false,
            },
            new()
            {
                ExecutionMode = Microsoft.ML.OnnxRuntime.ExecutionMode.ORT_PARALLEL,
            });
    }

    [Benchmark]
    public IReadOnlyCollection<FaceDetectorResult> First() => _scrfd1.Detect(_imgTensor, new Size(640, 640), 1.0f);

    [Benchmark]
    public IReadOnlyCollection<FaceDetectorResult> Second() => _scrfd2.Detect(_imgTensor, new Size(640, 640), 1.0f);
}
