// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using FaceAiSharp.Extensions;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceAiSharp.Tests;

public class ImageToTensorCorrectness
{
    private readonly string _imgPath = $@"{AppContext.BaseDirectory}../../../../examples/obama_family.jpg";
    private readonly Image<Rgb24> _img;

    public ImageToTensorCorrectness()
    {
        _img = Image.Load<Rgb24>(_imgPath);
    }

    [Fact]
    public void OptimizedEqualsNaive()
    {
        var optimized = _img.ToTensor();
        var naive = Naive(_img);

        var optimizedBuffer = optimized.Buffer.Span;
        var naiveBuffer = naive.Buffer.Span;
        Assert.Equal(naiveBuffer.Length, optimizedBuffer.Length);

        for (var i = 0; i < naiveBuffer.Length; i++)
        {
            Assert.Equal(naiveBuffer[i], optimizedBuffer[i], 0.00001f);
        }
    }

    [Fact]
    public void ArcFaceOptimizedEqualsNaive()
    {
        _img.EnsureProperlySizedDestructive(new() { Size = new(112, 112), PadColor = Color.Black, Mode = SixLabors.ImageSharp.Processing.ResizeMode.Pad }, false);
        var optimized = ArcFaceEmbeddingsGenerator.CreateImageTensor(_img);
        var naive = ArcFaceNaive(_img);

        var optimizedBuffer = optimized.Buffer.Span;
        var naiveBuffer = naive.Buffer.Span;
        Assert.Equal(naiveBuffer.Length, optimizedBuffer.Length);

        for (var i = 0; i < naiveBuffer.Length; i++)
        {
            Assert.Equal(naiveBuffer[i], optimizedBuffer[i], 0.00001f);
        }
    }

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

    private static DenseTensor<float> ArcFaceNaive(Image<Rgb24> img)
    {
        // originally was
        // var ret = new DenseTensor<float>(new[] { 1, 3, 112, 112 });
        var ret = new DenseTensor<float>(new[] { 1, 3, img.Height, img.Width });

        img.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < accessor.Height; y++)
            {
                Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                for (var x = 0; x < accessor.Width; x++)
                {
                    ret[0, 0, y, x] = pixelSpan[x].R;
                    ret[0, 1, y, x] = pixelSpan[x].G;
                    ret[0, 2, y, x] = pixelSpan[x].B;
                }
            }
        });

        return ret;
    }
}
