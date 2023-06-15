// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceAiSharp;

public interface IFaceEmbeddingsGenerator
{
    /// <summary>
    ///     Generates an embedding vector for the given aligned face image. The embedding vectors produced by this function
    ///     can be used to compare faces; vectors for images of the same person will be closer to each other in the embedding space.
    ///     The choice of similarity metric for comparing the vectors depends on the specific implementation and the nature of the problem.
    ///     Common similarity metrics include cosine similarity and euclidean distance.
    /// </summary>
    /// <remarks>
    ///     Depending on the specific face recognition implementation, it may be important to properly align the input image. How the
    ///     alignment is done depends on the model used for the implementation.
    /// </remarks>
    /// <param name="alignedFace">An aligned image of a face.</param>
    /// <returns>
    ///     An embedding vector (in a high-dimensional space) that represents the unique features of the given face. This vector can be
    ///     compared with others to determine if multiple faces belong to the same person.
    /// </returns>
    float[] GenerateEmbedding(Image<Rgb24> alignedFace);

    /// <summary>
    ///     Aligns the given face image based on the provided facial landmarks. Alignment is specific to the model used for implementation.
    ///     This method operates in-place and thus modifies the given image.
    /// </summary>
    /// <param name="face">The image of the face to be aligned. The given image will be mutated.</param>
    /// <param name="landmarks">
    ///     A read-only list of points representing facial landmarks. These landmarks usually include the coordinates of features like eyes,
    ///     nose, mouth, etc. Which landmarks are expected in which order depends on the implementation.
    /// </param>
    void AlignFaceUsingLandmarks(Image<Rgb24> face, IReadOnlyList<PointF> landmarks);
}
