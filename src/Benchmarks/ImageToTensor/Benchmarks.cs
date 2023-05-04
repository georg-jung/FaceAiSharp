// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using BenchmarkDotNet.Attributes;
using Benchmarks.ImageToTensor;
using FaceAiSharp;
using FaceAiSharp.Extensions;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Benchmarks.ImageToTensor;

[MemoryDiagnoser]
public class Benchmarks
{
    private readonly Image _img = Image.Load(@"C:\Users\georg\facePics\avGroup.jpg");
    private readonly Image<Rgb24> _preprocImg;
    private readonly Image<RgbaVector> _preprocImgRgbaVector;

    public Benchmarks()
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
        var x2 = x.Image.CloneAs<RgbaVector>();

        _preprocImg = x.Image;
        _preprocImgRgbaVector = x2;
    }

    /*
    [Benchmark(Baseline = true)]
    public DenseTensor<float> Naive() => Naive(_preprocImg);

    [Benchmark]
    public DenseTensor<float> Vector3() => Vector3(_preprocImg);

    [Benchmark]
    public DenseTensor<float> ProcessPixelRowsAsVector4() => ProcessPixelRowsAsVector4(_preprocImg);

    // don't run this every time, it is really close to ProcessPixelRowsAsVector4
    // [Benchmark]
    public DenseTensor<float> ProcessPixelRowsAsVector4RgbaVector() => ProcessPixelRowsAsVector4RgbaVector(_preprocImgRgbaVector);
    */

    [Benchmark]
    public DenseTensor<float> OptimizedBySkywalkerisnull() => OptimizedBySkywalkerisnull(_preprocImg);

    [Benchmark]
    public DenseTensor<float> OptimizedBySkywalkerisnullV2() => OptimizedBySkywalkerisnullV2(_preprocImg);

    [Benchmark]
    public DenseTensor<float> Production() => _preprocImg.ToTensor();

    private static DenseTensor<float> Naive(Image<Rgb24> img)
    {
        var ret = new DenseTensor<float>(new[] { 1, 3, img.Height, img.Width });

        var mean = new[] { 0.5f, 0.5f, 0.5f };
        img.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < accessor.Height; y++)
            {
                Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                for (var x = 0; x < accessor.Width; x++)
                {
                    ret[0, 0, y, x] = (pixelSpan[x].R / 255f) - mean[0];
                    ret[0, 1, y, x] = (pixelSpan[x].G / 255f) - mean[1];
                    ret[0, 2, y, x] = (pixelSpan[x].B / 255f) - mean[2];
                }
            }
        });

        return ret;
    }

    private static DenseTensor<float> Vector3(Image<Rgb24> img)
    {
        var ret = new DenseTensor<float>(new[] { 1, 3, img.Height, img.Width });

        var mean = new Vector3(0.5f);
        var max = new Vector3(byte.MaxValue);

        img.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < accessor.Height; y++)
            {
                Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);

                // pixelSpan.Length allows optimizations over accessor.Width but is semantically equivalent
                // see https://docs.sixlabors.com/articles/imagesharp/pixelbuffers.html
                for (var x = 0; x < pixelSpan.Length; x++)
                {
                    ref var val = ref pixelSpan[x];
                    var pxVec = new Vector3(val.R, val.G, val.B);
                    pxVec = (pxVec / max) - mean;
                    ret[0, 0, y, x] = pxVec.X;
                    ret[0, 1, y, x] = pxVec.Y;
                    ret[0, 2, y, x] = pxVec.Z;
                }
            }
        });

        return ret;
    }

    private static DenseTensor<float> ProcessPixelRowsAsVector4(Image<Rgb24> img)
    {
        var ret = new DenseTensor<float>(new[] { 1, 3, img.Height, img.Width });

        var mean = new Vector4(0.5f);
        var max = new Vector4(byte.MaxValue);

        img.Mutate(op => op.ProcessPixelRowsAsVector4((row, z) =>
        {
            for (int x = 0; x < row.Length; x++)
            {
                var y = z.Y;
                var pxVec = row[x] - mean;
                ret[0, 0, y, x] = pxVec.X;
                ret[0, 1, y, x] = pxVec.Y;
                ret[0, 2, y, x] = pxVec.Z;
            }
        }));

        return ret;
    }

    private static DenseTensor<float> ProcessPixelRowsAsVector4RgbaVector(Image<RgbaVector> img)
    {
        var ret = new DenseTensor<float>(new[] { 1, 3, img.Height, img.Width });

        var mean = new Vector4(0.5f);
        var max = new Vector4(byte.MaxValue);

        img.Mutate(op => op.ProcessPixelRowsAsVector4((row, z) =>
        {
            for (int x = 0; x < row.Length; x++)
            {
                var y = z.Y;
                var pxVec = row[x] - mean;
                ret[0, 0, y, x] = pxVec.X;
                ret[0, 1, y, x] = pxVec.Y;
                ret[0, 2, y, x] = pxVec.Z;
            }
        }));

        return ret;
    }

    private static DenseTensor<float> OptimizedBySkywalkerisnull(Image<Rgb24> img)
    {
        var mean = new[] { 0.5f, 0.5f, 0.5f };
        var stddev = new[] { 1f, 1f, 1f };
        var dims = new[] { 1, 3, 640, 640 };
        return BySkywalkerisnull.ToTensor(img, mean, stddev, dims);
    }

    private static DenseTensor<float> OptimizedBySkywalkerisnullV2(Image<Rgb24> img)
    {
        var mean = new[] { 0.5f, 0.5f, 0.5f };
        var stddev = new[] { 1f, 1f, 1f };
        var dims = new[] { 1, 3, 640, 640 };
        return BySkywalkerisnullV2.ToTensor(img, mean, stddev, dims);
    }
}
