// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using FaceAiSharp.Abstractions;
using FaceAiSharp.Extensions;
using SixLabors.ImageSharp;

namespace FaceAiSharp;

public sealed class FaceOnnxLandmarkExtractor : IFaceLandmarksExtractor, IDisposable
{
    private readonly FaceONNX.FaceLandmarksExtractor _fonnx = new();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float GetFaceAlignmentAngle(IReadOnlyCollection<Point> landmarks)
        => FaceONNX.Landmarks.GetRotationAngle(landmarks.Select(p => new System.Drawing.Point(p.X, p.Y)).ToArray());

    public void Dispose() => _fonnx.Dispose();

    public IReadOnlyCollection<Point> Detect(Image image)
    {
        var img = image.ToFaceOnnxFloatArray();
        var res = _fonnx.Forward(img);

        return res.Select(p => new Point(p.X, p.Y)).ToList();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    float IFaceLandmarksExtractor.GetFaceAlignmentAngle(IReadOnlyCollection<Point> landmarks)
        => GetFaceAlignmentAngle(landmarks);
}
