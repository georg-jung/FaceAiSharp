// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics.Tensors;
using BenchmarkDotNet.Attributes;
using NumSharp;
using SixLabors.ImageSharp;

namespace Benchmarks;

[MemoryDiagnoser]
/* [ShortRunJob] */
public class ScrfdGenerateAnchorCenters
{
    private const int Height = 640;
    private const int Width = 960;
    private const int Stride = 16;
    private const int NumAnchors = 2;

    /* 2025-10-25:
    BenchmarkDotNet v0.14.0, Windows 11 (10.0.26100.6899)
    Unknown processor
    .NET SDK 9.0.305
      [Host]     : .NET 8.0.21 (8.0.2125.47513), X64 RyuJIT AVX2
      DefaultJob : .NET 8.0.21 (8.0.2125.47513), X64 RyuJIT AVX2


    | Method                            | Mean         | Error      | StdDev     | Median       | Gen0     | Gen1   | Allocated  |
    |---------------------------------- |-------------:|-----------:|-----------:|-------------:|---------:|-------:|-----------:|
    | GenerateAnchorCentersNaive        |     5.423 us |  0.1005 us |  0.2631 us |     5.310 us |   4.5624 |      - |   37.52 KB |
    | GenerateAnchorCentersSimdMultiply |     5.490 us |  0.0993 us |  0.2474 us |     5.387 us |   4.5624 |      - |   37.52 KB |
    | GenerateAnchorCentersNumSharp     | 1,385.014 us | 27.4523 us | 60.2586 us | 1,351.456 us | 376.9531 | 9.7656 | 3074.64 KB |
    */

    [Benchmark]
    public void GenerateAnchorCentersNaive()
    {
        GenerateAnchorCentersNaive(new Size(Width, Height), Stride, NumAnchors);
    }

    [Benchmark]
    public void GenerateAnchorCentersSimdMultiply()
    {
        GenerateAnchorCentersSimdMultiply(new Size(Width, Height), Stride, NumAnchors);
    }

    [Benchmark]
    public void GenerateAnchorCentersNumSharp()
    {
        GenerateAnchorCentersNumSharp(new Size(Width, Height), Stride, NumAnchors);
    }

    private static Span<float> GenerateAnchorCentersNaive(Size inputSize, int stride, int numAnchors)
    {
        var height = inputSize.Height / stride;
        var width = inputSize.Width / stride;
        var data = new float[height * width * numAnchors * 2];
        var idx = 0;

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                for (var a = 0; a < numAnchors; a++)
                {
                    data[idx++] = x * stride;
                    data[idx++] = y * stride;
                }
            }
        }

        return data;
    }

    private static Span<float> GenerateAnchorCentersSimdMultiply(Size inputSize, int stride, int numAnchors)
    {
        var height = inputSize.Height / stride;
        var width = inputSize.Width / stride;
        var data = new float[height * width * numAnchors * 2];
        var idx = 0;

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                for (var a = 0; a < numAnchors; a++)
                {
                    data[idx++] = x;
                    data[idx++] = y;
                }
            }
        }

        TensorPrimitives.Multiply(data, stride, data);
        return data;
    }

    private static NDArray GenerateAnchorCentersNumSharp(Size inputSize, int stride, int numAnchors)
    {
        // translated from https://github.com/deepinsight/insightface/blob/f091989568cad5a0244e05be1b8d58723de210b0/detection/scrfd/tools/scrfd.py#L185
        var height = inputSize.Height / stride;
        var width = inputSize.Width / stride;
        var (mgrid1, mgrid2) = np.mgrid(np.arange(height), np.arange(width));
        var anchorCenters = np.stack([mgrid2, mgrid1], axis: -1).astype(np.float32);
        anchorCenters = (anchorCenters * stride).reshape(-1, 2);
        if (numAnchors > 1)
        {
            anchorCenters = np.stack([anchorCenters, anchorCenters], axis: 1).reshape(-1, 2);
        }

        return anchorCenters;
    }
}
