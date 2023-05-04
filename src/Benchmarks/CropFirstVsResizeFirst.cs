// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Attributes;
using FaceAiSharp.Extensions;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

namespace Benchmarks;

[MemoryDiagnoser]
public class CropFirstVsResizeFirst
{
    private readonly Image _img = Image.Load(@"C:\Users\georg\OneDrive\Bilder\20160111-0162_GJ.jpg");
    private readonly Rectangle _crop = new(20, 50, 1000, 2000);

    public static Image ResizeFirst(Image sourceImage, Rectangle sourceArea, int extractedMaxEdgeSize)
        => sourceImage.Clone(op =>
        {
            var longestDim = Math.Max(sourceArea.Width, sourceArea.Height);
            var toLargeFactor = Math.Max(1.0, longestDim / (double)extractedMaxEdgeSize);
            var factor = 1.0 / toLargeFactor; // scale factor

            if (factor < 1)
            {
                var curSize = op.GetCurrentSize();
                op.Resize(curSize.Scale(factor));
                sourceArea = sourceArea.Scale(factor);
            }

            op.Crop(sourceArea);
        });

    public static Image CropFirst(Image sourceImage, Rectangle sourceArea, int extractedMaxEdgeSize)
    => sourceImage.Clone(op =>
    {
        var longestDim = Math.Max(sourceArea.Width, sourceArea.Height);
        var toLargeFactor = Math.Max(1.0, longestDim / (double)extractedMaxEdgeSize);
        var factor = 1.0 / toLargeFactor; // scale factor

        op.Crop(sourceArea);

        if (factor < 1)
        {
            var curSize = op.GetCurrentSize();
            op.Resize(curSize.Scale(factor));
        }
    });

    [Benchmark]
    public Image ResizeFirst() => ResizeFirst(_img, _crop, 250);

    [Benchmark]
    public Image CropFirst() => CropFirst(_img, _crop, 250);
}
