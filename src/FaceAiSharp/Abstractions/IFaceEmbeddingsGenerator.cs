// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceAiSharp.Abstractions;

public interface IFaceEmbeddingsGenerator
{
    /// <summary>
    ///     Generates vectors that are closer to other vectors returned by this function if the given images belong to the same person.
    ///     The best similarity metric to determine closeness depends on that implementation.
    ///     Typical metrics are Cosine Similarity and Euclidean Similarity.
    /// </summary>
    /// <param name="alignedFace">An aligned image of a face.</param>
    /// <returns>An embedding vector that correspond to the given face.</returns>
    public float[] Generate(Image<Rgb24> alignedFace);
}
