// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceAiSharp.Extensions;

internal static class ImageToTensorExtensions
{
    /* this is heavily based on https://github.com/skywalkerisnull/onnxruntime-csharp-cv-template/blob/bb1454a51a722e293a918d2b5a25abda864f9f74/utils/ImageHelper.cs#L62
     * see also https://github.com/SixLabors/ImageSharp/discussions/1955
     */

    /// <summary>
    /// Efficiently converts one image to an input tensor.
    /// Preprocesses for {r, g, b}: <c>r = (r * / 255) - 0.5</c>.
    /// </summary>
    /// <param name="image">The image to preprocess.</param>
    /// <returns>A tensor containing the preprocessed image.</returns>
    public static DenseTensor<float> ToTensor(this Image<Rgb24> image)
    {
        var mean = new[] { 0.5f, 0.5f, 0.5f };
        var stddev = new[] { 1f, 1f, 1f };
        var dims = new[] { 1, 3, image.Height, image.Width };
        return ImageToTensor(new[] { image }, mean, stddev, dims);
    }

    /// <summary>
    /// Efficiently converts one image to an input tensor.
    /// Preprocesses for {r, g, b}: <c>r = (r * (1 / 255 * stddev.r)) - (mean.r / stddev.r)</c>.
    /// Thus, if you need simple r = r, you could pass stddev = 1/255 and mean = 0. Note that
    /// while it might seem unnecessary to use this method then, it's memory access
    /// is optimized and the conversion is almost two orders of magnitude faster than a simple
    /// approach.
    /// </summary>
    /// <param name="image">The image to preprocess.</param>
    /// <param name="mean">The rgb mean values used for normalization.</param>
    /// <param name="stddev">The rgb stddev values used for normalization.</param>
    /// <returns>A tensor containing the preprocessed image.</returns>
    public static DenseTensor<float> ToTensor(this Image<Rgb24> image, float[] mean, float[] stddev)
    {
        var dims = new[] { 1, 3, image.Height, image.Width };
        return ImageToTensor(new[] { image }, mean, stddev, dims);
    }

    /// <summary>
    /// Efficiently converts one image to an input tensor.
    /// Preprocesses for {r, g, b}: <c>r = (r * (1 / 255 * stddev.r)) - (mean.r / stddev.r)</c>.
    /// Thus, if you need simple r = r, you could pass stddev = 1/255 and mean = 0. Note that
    /// while it might seem unnecessary to use this method then, it's memory access
    /// is optimized and the conversion is almost two orders of magnitude faster than a simple
    /// approach.
    /// </summary>
    /// <param name="image">The image to preprocess.</param>
    /// <param name="mean">The rgb mean values used for normalization.</param>
    /// <param name="stddev">The rgb stddev values used for normalization.</param>
    /// <param name="inputDimension">
    ///     The dimensions the created tensor should have. Might throw exceptions if
    ///     memory access fails due to invalid dimensions. The first dimension's value
    ///     will be set to 1 no matter what you pass in, as this method processes
    ///     exactly one picture.
    /// </param>
    /// <param name="convertToBgr">Swap the image's channels during conversion. RGB -> BGR. Some models expect their input data in BGR instead of RGB.</param>
    /// <returns>A tensor containing the preprocessed image.</returns>
    public static DenseTensor<float> ToTensor(this Image<Rgb24> image, float[] mean, float[] stddev, int[] inputDimension, bool convertToBgr = false)
    {
        return ImageToTensor(new[] { image }, mean, stddev, inputDimension, convertToBgr);
    }

    /// <summary>
    /// Efficiently converts images to an input tensor for batch processing.
    /// Preprocesses for {r, g, b}: <c>r = (r * (1 / 255 * stddev.r)) - (mean.r / stddev.r)</c>.
    /// Thus, if you need simple r = r, you could pass stddev = 1/255 and mean = 0.
    /// </summary>
    /// <param name="images">The images to convert.</param>
    /// <param name="mean">The rgb mean values used for normalization.</param>
    /// <param name="stddev">The rgb stddev values used for normalization.</param>
    /// <param name="inputDimension">The size of the tensor that the OnnxRuntime model is expecting, e.g. [1, 3, 224, 224].</param>
    /// <param name="convertToBgr">Swap the image's channels during conversion. RGB -> BGR. Some models expect their input data in BGR instead of RGB.</param>
    /// <returns>A tensor that contains the converted batch of images.</returns>
    public static DenseTensor<float> ImageToTensor(IReadOnlyCollection<Image<Rgb24>> images, float[] mean, float[] stddev, int[] inputDimension, bool convertToBgr = false)
    {
        // Calculate these outside the loop
        var normR = mean[0] / stddev[0];
        var normG = mean[1] / stddev[1];
        var normB = mean[2] / stddev[2];

        var stdNormR = 1 / (255f * stddev[0]);
        var stdNormG = 1 / (255f * stddev[1]);
        var stdNormB = 1 / (255f * stddev[2]);

        inputDimension[0] = images.Count;

        var input = new DenseTensor<float>(inputDimension);
        var imgCnt = 0;
        foreach (var image in images)
        {
            var imgStrides = input.Strides[0] * imgCnt;
            image.ProcessPixelRows(pixelAccessor =>
            {
                var strides = input.Strides;
                var inputSpan = input.Buffer.Span;
                for (var y = 0; y < image.Height; y++)
                {
                    var index = imgStrides + (y * strides[2]);
                    var rowSpan = pixelAccessor.GetRowSpan(y);

                    // Faster indexing into the span
                    var spanR = inputSpan.Slice(index, rowSpan.Length);
                    index += strides[1];
                    var spanG = inputSpan.Slice(index, rowSpan.Length);
                    index += strides[1];
                    var spanB = inputSpan.Slice(index, rowSpan.Length);

                    if (convertToBgr)
                    {
                        var xr = spanR;
                        spanR = spanB;
                        spanB = xr;
                    }

                    // Now we can just directly loop through and copy the values directly from span to span.
                    for (int x = 0; x < rowSpan.Length; x++)
                    {
                        spanR[x] = (rowSpan[x].R * stdNormR) - normR;
                        spanG[x] = (rowSpan[x].G * stdNormG) - normG;
                        spanB[x] = (rowSpan[x].B * stdNormB) - normB;
                    }
                }
            });
            imgCnt++;
        }

        return input;
    }
}
