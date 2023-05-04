// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

namespace FaceAiSharp.Extensions;

internal static class ImageExtensions
{
    internal static float[][,] ToFaceOnnxFloatArray(this Image image)
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
}
