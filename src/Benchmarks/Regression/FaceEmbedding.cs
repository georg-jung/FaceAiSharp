// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Attributes;
using FaceAiSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Benchmarks.Regression;

[BenchmarkCategory("regression")]
public class FaceEmbedding
{
    private ScrfdDetector _detector = null!;
    private ArcFaceEmbeddingsGenerator _embedder = null!;
    private Image<Rgb24> _portrait = null!;
    private Image<Rgb24> _aligned = null!;
    private IReadOnlyList<PointF> _landmarks = null!;

    [GlobalSetup]
    public void Setup()
    {
        _detector = new ScrfdDetector(new ScrfdDetectorOptions
        {
            ModelPath = BenchmarkData.ModelPath("scrfd_2.5g_kps.onnx"),
        });
        _embedder = new ArcFaceEmbeddingsGenerator(new ArcFaceEmbeddingsGeneratorOptions
        {
            ModelPath = BenchmarkData.ModelPath("arcfaceresnet100-11-int8.onnx"),
        });

        _portrait = BenchmarkData.LoadImage("Barack_Obama_03.jpg");
        var faces = _detector.DetectFaces(_portrait);
        if (faces.Count == 0)
        {
            throw new InvalidOperationException("Benchmark setup failed: no face was detected in Barack_Obama_03.jpg.");
        }

        var face = faces.MaxBy(x => x.Confidence);
        _landmarks = face.Landmarks
            ?? throw new InvalidOperationException("Benchmark setup failed: the detected face has no landmarks.");

        _aligned = _portrait.Clone();
        ArcFaceEmbeddingsGenerator.AlignFaceUsingLandmarks(_aligned, _landmarks);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _detector.Dispose();
        _embedder.Dispose();
        _portrait.Dispose();
        _aligned.Dispose();
    }

    /// <summary>
    /// Face alignment based on landmarks. AlignFaceUsingLandmarks mutates its input,
    /// so this includes cloning the source image.
    /// </summary>
    [Benchmark]
    public void CloneAndAlign()
    {
        using var clone = _portrait.Clone();
        ArcFaceEmbeddingsGenerator.AlignFaceUsingLandmarks(clone, _landmarks);
    }

    /// <summary>
    /// Embedding generation for a pre-aligned face: tensor conversion, inference and normalization.
    /// </summary>
    /// <returns>The embedding vector.</returns>
    [Benchmark]
    public float[] GenerateEmbedding() => _embedder.GenerateEmbedding(_aligned);
}
