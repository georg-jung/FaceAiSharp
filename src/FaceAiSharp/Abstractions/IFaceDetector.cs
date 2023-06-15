// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceAiSharp;

public interface IFaceDetector
{
    /// <summary>
    ///     Detects faces in the provided image. When supported by the implementation, facial landmark points are also extracted for each detected face.
    ///     The number and interpretation of landmark points are specific to the implementation.
    /// </summary>
    /// <remarks>
    ///     - The quality of the input image can affect the accuracy of face detection. High-resolution images where faces are clearly visible tend to yield better results.
    ///     - Facial landmark points refer to specific features on a face, such as center points or corners of the eyes, nose tip, and mouth corners. These can be used for alignment, emotion analysis, etc.
    ///     - Bounding boxes represent the regions within the image where faces are detected.
    ///     - Confidence values indicate the probability that the detected region actually contains a face.
    ///     - It's advisable to consult the documentation of the specific implementation to understand how to interpret and use the landmark points effectively.
    /// </remarks>
    /// <param name="image">The input image which may contain one or more human faces.</param>
    /// <returns>
    ///     A collection of results, where each result contains the bounding box, facial landmark coordinates (if supported), and a confidence score for each face detected in the image.
    /// </returns>
    IReadOnlyCollection<FaceDetectorResult> DetectFaces(Image<Rgb24> image);
}

public readonly record struct FaceDetectorResult(RectangleF Box, IReadOnlyList<PointF>? Landmarks, float? Confidence);
