// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Attributes;
using FaceAiSharp;
using FaceAiSharp.Extensions;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Benchmarks.Regression;

[BenchmarkCategory("regression")]
public class FaceDetection
{
    private ScrfdDetector _detector = null!;
    private Image<Rgb24> _groupPhoto = null!;
    private Image<Rgb24> _preprocessed640 = null!;
    private IDisposable? _preprocessed640Disposable;
    private DenseTensor<float> _tensor640 = null!;

    [GlobalSetup]
    public void Setup()
    {
        _detector = new ScrfdDetector(new ScrfdDetectorOptions
        {
            ModelPath = BenchmarkData.ModelPath("scrfd_2.5g_kps.onnx"),
        });

        _groupPhoto = BenchmarkData.LoadImage("obama_family.jpg");

        /* EnsureProperlySized returns the original instance (and no disposable) if the image
         * already has the requested dimensions, so dispose the returned disposable instead of
         * the image to avoid a double dispose of _groupPhoto. */
        var (img, disposable) = _groupPhoto.EnsureProperlySized<Rgb24>(
            new ResizeOptions
            {
                Size = new Size(640),
                Position = AnchorPositionMode.TopLeft,
                Mode = ResizeMode.BoxPad,
                PadColor = Color.Black,
            },
            false);
        _preprocessed640 = img;
        _preprocessed640Disposable = disposable;
        _tensor640 = ScrfdDetector.CreateImageTensor(img);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _detector.Dispose();
        _preprocessed640Disposable?.Dispose();
        _groupPhoto.Dispose();
    }

    /// <summary>
    /// End-to-end detection including resizing, tensor conversion, inference and postprocessing.
    /// </summary>
    /// <returns>The detected faces.</returns>
    [Benchmark]
    public IReadOnlyCollection<FaceDetectorResult> DetectFaces() => _detector.DetectFaces(_groupPhoto);

    /// <summary>
    /// Inference and postprocessing only, based on a precomputed input tensor.
    /// </summary>
    /// <returns>The detected faces.</returns>
    [Benchmark]
    public IReadOnlyCollection<FaceDetectorResult> InferAndPostprocess() => _detector.Detect(_tensor640, new Size(640, 640), 1.0f);

    /// <summary>
    /// Preprocessing only: convert a 640x640 image to an input tensor.
    /// </summary>
    /// <returns>The input tensor.</returns>
    [Benchmark]
    public DenseTensor<float> ImageToTensor() => _preprocessed640.ToTensor();
}
