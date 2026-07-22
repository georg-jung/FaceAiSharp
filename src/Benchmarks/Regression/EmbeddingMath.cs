// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Attributes;
using FaceAiSharp.Extensions;

namespace Benchmarks.Regression;

[BenchmarkCategory("regression")]
public class EmbeddingMath
{
    private const int EmbeddingLength = 512;
    private float[] _x = null!;
    private float[] _y = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rnd = new Random(42);
        _x = CreateVector(rnd);
        _y = CreateVector(rnd);
    }

    [Benchmark]
    public float Dot() => _x.Dot(_y);

    [Benchmark]
    public float CosineSimilarity() => _x.CosineSimilarity(_y);

    [Benchmark]
    public float EuclideanDistance() => _x.EuclideanDistance(_y);

    [Benchmark]
    public float[] ToUnitLength() => GeometryExtensions.ToUnitLength(_x);

    private static float[] CreateVector(Random rnd)
    {
        var vec = new float[EmbeddingLength];
        for (var i = 0; i < vec.Length; i++)
        {
            vec[i] = (float)((rnd.NextDouble() * 2) - 1);
        }

        return vec;
    }
}
