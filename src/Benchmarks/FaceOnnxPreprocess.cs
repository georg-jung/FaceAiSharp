// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Benchmarks;

public static class FaceOnnxPreprocess
{
    public static float[][,] ToFaceOnnxFloatArrayParallel(this Image image)
    {
        var r = new float[image.Height, image.Width];
        var g = new float[image.Height, image.Width];
        var b = new float[image.Height, image.Width];
        image.Mutate(c => c.ProcessPixelRowsAsVector4((row, point) =>
        {
            for (var x = 0; x < row.Length; x++)
            {
                // Get a reference to the pixel at position x
                ref var pixel = ref row[x];
                var y = point.Y;
                r[y, x] = pixel.X;
                g[y, x] = pixel.Y;
                b[y, x] = pixel.Z;
            }
        }));
        return new float[][,] { b, g, r };
    }

    public static float[][,] ToFaceOnnxFloatArray(this Image<RgbaVector> image)
    {
        var r = new float[image.Height, image.Width];
        var g = new float[image.Height, image.Width];
        var b = new float[image.Height, image.Width];
        image.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < accessor.Height; y++)
            {
                var pixelRow = accessor.GetRowSpan(y);

                // pixelRow.Length has the same value as accessor.Width,
                // but using pixelRow.Length allows the JIT to optimize away bounds checks:
                for (var x = 0; x < pixelRow.Length; x++)
                {
                    // Get a reference to the pixel at position x
                    ref var pixel = ref pixelRow[x];
                    r[y, x] = pixel.R;
                    g[y, x] = pixel.G;
                    b[y, x] = pixel.B;
                }
            }
        });
        return new float[][,] { b, g, r };
    }
}
