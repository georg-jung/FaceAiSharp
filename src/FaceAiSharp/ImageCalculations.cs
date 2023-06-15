// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using FaceAiSharp.Extensions;
using SixLabors.ImageSharp;

namespace FaceAiSharp;

public static class ImageCalculations
{
    /// <summary>
    ///     Estimates bounding boxes for eyes if just their center points are known. It assumes the size of the required boxes
    ///     linearly depends on the distance of the eye center points.
    /// </summary>
    /// <param name="leftEyeCenter">The center point of the left eye.</param>
    /// <param name="rightEyeCenter">The center point of the right eye.</param>
    /// <param name="distanceDivisor">
    ///     The higher the value given is, the smaller the boxes around the eyes will be.
    ///     Square eye box length = distance between eyes / <paramref name="distanceDivisor"/> * 2.
    ///     3.0 seems to be a suitable value for typical eye sizes and some additional space around the eyes.
    /// </param>
    /// <returns>Calculated bounding boxes for the eyes.</returns>
    public static EyeBoxes GetEyeBoxesFromCenterPoints(PointF leftEyeCenter, PointF rightEyeCenter, float distanceDivisor = 3)
    {
        var dist = leftEyeCenter.EuclideanDistance(rightEyeCenter);
        var squareAroundEyeLen = dist / distanceDivisor;
        var eyeRectSz = new Size((int)squareAroundEyeLen * 2);
        leftEyeCenter.Offset(-squareAroundEyeLen, -squareAroundEyeLen);
        rightEyeCenter.Offset(-squareAroundEyeLen, -squareAroundEyeLen);
        var leyeRect = new Rectangle(Point.Round(leftEyeCenter), eyeRectSz);
        var reyeRect = new Rectangle(Point.Round(rightEyeCenter), eyeRectSz);
        return new(leyeRect, reyeRect);
    }

    public readonly record struct EyeBoxes(Rectangle Left, Rectangle Right);
}
