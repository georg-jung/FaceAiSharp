// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using SixLabors.ImageSharp;

namespace FaceAiSharp.Abstractions;

public interface IFaceLandmarksExtractor
{
    /// <summary>
    /// Extract facial landmarks point from a properly cropped image of a face. No prior alignment needed.
    /// The amount and meaning of the points depends on the model used for implementation.
    /// </summary>
    /// <param name="image">Cropped image of a face.</param>
    /// <returns>Facial landmark coordiantes.</returns>
    IReadOnlyCollection<Point> Detect(Image image);

    float GetFaceAlignmentAngle(IReadOnlyCollection<Point> landmarks);
}
