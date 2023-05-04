// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Benchmarks.ImageToTensor;

[System.Diagnostics.CodeAnalysis.SuppressMessage("StyleCop.CSharp.DocumentationRules", "SA1614:Element parameter documentation should have text", Justification = "These are just benchmarks")]
[System.Diagnostics.CodeAnalysis.SuppressMessage("StyleCop.CSharp.DocumentationRules", "SA1615:Element return value should be documented", Justification = "These are just benchmarks")]
internal static class BySkywalkerisnullV2
{
    /* this is heavily based on https://github.com/skywalkerisnull/onnxruntime-csharp-cv-template/blob/bb1454a51a722e293a918d2b5a25abda864f9f74/utils/ImageHelper.cs#L62
     * see also https://github.com/SixLabors/ImageSharp/discussions/1955
     */

    public static DenseTensor<float> ToTensor(Image<Rgb24> image, float[] mean, float[] stddev, int[] inputDimension)
    {
        return ImageToTensor(new[] { image }, mean, stddev, inputDimension).First();
    }

    /// <summary>
    /// Converts the list of images into batches and list of input tensors.
    /// </summary>
    /// <param name="images"></param>
    /// <param name="mean"></param>
    /// <param name="stddev"></param>
    /// <param name="inputDimension">The size of the tensor that the OnnxRuntime model is expecting, e.g. [1, 3, 224, 224].</param>
    public static List<DenseTensor<float>> ImageToTensor(IReadOnlyList<Image<Rgb24>> images, float[] mean, float[] stddev, int[] inputDimension)
    {
        // Used to create more than one batch
        int numberBatches = 1;

        // If required, can create batches of different sizes
        var batchSizes = new int[] { images.Count };

        var strides = GetStrides(inputDimension);

        var inputs = new List<DenseTensor<float>>();

        // Faster normalisation process
        var normR = mean[0] / stddev[0];
        var normG = mean[1] / stddev[1];
        var normB = mean[2] / stddev[2];

        var stdNormR = 1 / (255f * stddev[0]);
        var stdNormG = 1 / (255f * stddev[1]);
        var stdNormB = 1 / (255f * stddev[2]);

        for (var j = 0; j < numberBatches; j++)
        {
            inputDimension[0] = batchSizes[j];

            // Need to directly use a DenseTensor here because we need access to the underlying span.
            DenseTensor<float> input = new DenseTensor<float>(inputDimension);

            for (var i = 0; i < batchSizes[j]; i++)
            {
                var image = images[i];

                image.Mutate(op => op.ProcessPixelRowsAsVector4((row, pos) =>
                {
                    var inputSpan = input.Buffer.Span;
                    var y = pos.Y;
                    var index = y * strides[2];

                    // Faster indexing into the span
                    var spanR = inputSpan.Slice(index, image.Width);
                    index += strides[1];
                    var spanG = inputSpan.Slice(index, image.Width);
                    index += strides[1];
                    var spanB = inputSpan.Slice(index, image.Width);

                    // Now we can just directly loop through and copy the values directly from span to span.
                    for (int x = 0; x < image.Width; x++)
                    {
                        spanR[x] = (row[x].W * stdNormR) - normR;
                        spanG[x] = (row[x].X * stdNormG) - normG;
                        spanB[x] = (row[x].Y * stdNormB) - normB;
                    }
                }));

                inputs.Add(input);
            }
        }

        return inputs;
    }

    /// <summary>
    /// Gets the set of strides that can be used to calculate the offset of n-dimensions in a 1-dimensional layout.
    /// </summary>
    private static int[] GetStrides(ReadOnlySpan<int> dimensions, bool reverseStride = false)
    {
        int[] strides = new int[dimensions.Length];

        if (dimensions.Length == 0)
        {
            return strides;
        }

        int stride = 1;
        if (reverseStride)
        {
            for (int i = 0; i < strides.Length; i++)
            {
                strides[i] = stride;
                stride *= dimensions[i];
            }
        }
        else
        {
            for (int i = strides.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= dimensions[i];
            }
        }

        return strides;
    }

    /// <summary>
    /// Calculates the 1-d index for n-d indices in layout specified by strides.
    /// </summary>
    private static int GetIndex(int[] strides, ReadOnlySpan<int> indices, int startFromDimension = 0)
    {
        Debug.Assert(strides.Length == indices.Length, "strides.Length must equal indices.Length");

        int index = 0;
        for (int i = startFromDimension; i < indices.Length; i++)
        {
            index += strides[i] * indices[i];
        }

        return index;
    }
}
