// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using FaceAiSharp.Extensions;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceAiSharp;

public sealed class FaceOnnxLandmarkExtractor : IFaceLandmarksDetector, IDisposable
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

    public IReadOnlyList<PointF> DetectLandmarks(Image<Rgb24> image)
    {
        throw new NotImplementedException();
    }

    public PointF GetLeftEyeCenter(IReadOnlyList<PointF> landmarks) => GetEyeCenter(landmarks, FaceONNX.Landmarks.GetLeftEye);

    public PointF GetRightEyeCenter(IReadOnlyList<PointF> landmarks) => GetEyeCenter(landmarks, FaceONNX.Landmarks.GetRightEye);

    private static PointF GetEyeCenter(IReadOnlyList<PointF> landmarks, Func<System.Drawing.Point[], System.Drawing.Point[]> eyeSelector)
    {
        var eye = FaceONNX.Landmarks.GetMeanPoint(eyeSelector(landmarks.Select(Point.Round).Select(p => new System.Drawing.Point(p.X, p.Y)).ToArray()));
        return new PointF(eye.X, eye.Y);
    }
}
