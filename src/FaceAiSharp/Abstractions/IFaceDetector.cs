// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using SixLabors.ImageSharp;

namespace FaceAiSharp.Abstractions;

public interface IFaceDetector
{
    /// <summary>
    /// Detect faces in an image. If the implementation supports extracting landmark points,
    /// it does so for each face in one step. The amount and meaning of the landmark points
    /// depends on the implementation.
    /// </summary>
    /// <param name="image">Image possibly containing one or more human faces.</param>
    /// <returns>Bounding boxes, facial landmark coordiantes and confidence for faces found in the given image.</returns>
    IReadOnlyCollection<FaceDetectorResult> Detect(Image image);

    /// <summary>
    /// Only applies if this implementation supports landmarks extraction, throws otherwise.
    /// Calculate the angle of a face, given the landmark points returned by <see cref="Detect(Image)"/>.
    /// </summary>
    /// <param name="landmarks">Landmark points as returned by <see cref="Detect(Image)"/>.</param>
    /// <returns>An angle by which the image of the face needs to be rotated to be aligned (both eyes on a horizontal line).</returns>
    /// <exception cref="NotSupportedException">This implementation does not support landmarks extraction.</exception>
    float GetFaceAlignmentAngle(IReadOnlyList<PointF> landmarks);

    /// <summary>
    /// Returns the center point of the left eye, based on the given landmarks.
    /// </summary>
    /// <param name="landmarks">Landmark points as returned by <see cref="Detect(Image)"/>.</param>
    /// <returns>Center point of the left eye.</returns>
    PointF GetLeftEyeCenter(IReadOnlyList<PointF> landmarks);

    /// <summary>
    /// Returns the center point of the right eye, based on the given landmarks.
    /// </summary>
    /// <param name="landmarks">Landmark points as returned by <see cref="Detect(Image)"/>.</param>
    /// <returns>Center point of the right eye.</returns>
    PointF GetRightEyeCenter(IReadOnlyList<PointF> landmarks);
}

public readonly record struct FaceDetectorResult(RectangleF Box, IReadOnlyList<PointF>? Landmarks, float? Confidence);
