// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceAiSharp;

public interface IFaceLandmarksDetector
{
    /// <summary>
    /// Extract facial landmark points from a properly cropped image of a face. The amount and meaning of the points depends on the model used for implementation.
    /// </summary>
    /// <param name="image">Cropped image of a face.</param>
    /// <returns>Facial landmark coordinates, or an empty list if no face could be detected in the image.</returns>
    IReadOnlyList<PointF> DetectLandmarks(Image<Rgb24> image);

    /// <summary>
    /// Returns the center point of the left eye, based on the given landmarks.
    /// </summary>
    /// <param name="landmarks">Landmark points as returned by <see cref="DetectLandmarks(Image{Rgb24})"/>.</param>
    /// <returns>Center point of the left eye.</returns>
    /// <exception cref="ArgumentException"><paramref name="landmarks"/> doesn't contain the expected landmark points, e.g. because it is the empty result of <see cref="DetectLandmarks(Image{Rgb24})"/> for an image without a face.</exception>
    PointF GetLeftEyeCenter(IReadOnlyList<PointF> landmarks);

    /// <summary>
    /// Returns the center point of the right eye, based on the given landmarks.
    /// </summary>
    /// <param name="landmarks">Landmark points as returned by <see cref="DetectLandmarks(Image{Rgb24})"/>.</param>
    /// <returns>Center point of the right eye.</returns>
    /// <exception cref="ArgumentException"><paramref name="landmarks"/> doesn't contain the expected landmark points, e.g. because it is the empty result of <see cref="DetectLandmarks(Image{Rgb24})"/> for an image without a face.</exception>
    PointF GetRightEyeCenter(IReadOnlyList<PointF> landmarks);
}
