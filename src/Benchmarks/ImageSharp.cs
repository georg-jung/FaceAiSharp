// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Attributes;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Benchmarks;

[MemoryDiagnoser]
public class ImageSharp
{
    private readonly Image _img = Image.Load(Regression.BenchmarkData.TestDataPath("jpgs", "biden_7mpx.jpg"));
    private readonly Image<RgbaVector> _imgV = Image.Load<RgbaVector>(Regression.BenchmarkData.TestDataPath("jpgs", "biden_7mpx.jpg"));

    [Benchmark]
    public void Parallel() => _img.ToFaceOnnxFloatArrayParallel();

    [Benchmark]
    public void ParallelVecBased() => _img.ToFaceOnnxFloatArrayParallel();

    [Benchmark]
    public void SingleWithClone() => _img.CloneAs<RgbaVector>().ToFaceOnnxFloatArray();

    [Benchmark]
    public void SingleVecBased() => _imgV.ToFaceOnnxFloatArray();
}
