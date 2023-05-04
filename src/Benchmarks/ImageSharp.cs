// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Attributes;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Benchmarks;

[MemoryDiagnoser]
public class ImageSharp
{
    private readonly Image _img = Image.Load(@"C:\Users\georg\OneDrive\Bilder\20160111-0162_GJ.jpg");
    private readonly Image<RgbaVector> _imgV = Image.Load<RgbaVector>(@"C:\Users\georg\OneDrive\Bilder\20160111-0162_GJ.jpg");

    [Benchmark]
    public void Parallel() => _img.ToFaceOnnxFloatArrayParallel();

    [Benchmark]
    public void ParallelVecBased() => _img.ToFaceOnnxFloatArrayParallel();

    [Benchmark]
    public void SingleWithClone() => _img.CloneAs<RgbaVector>().ToFaceOnnxFloatArray();

    [Benchmark]
    public void SingleVecBased() => _imgV.ToFaceOnnxFloatArray();
}
