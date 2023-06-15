// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using FaceAiSharp.Extensions;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace FaceAiSharp;

public static class Applications
{
    /// <summary>
    /// Blurs all faces that are found by the given face detector. Modifies the given image (destructive).
    /// </summary>
    /// <param name="detector">The face detector to use.</param>
    /// <param name="input">The image to search and blur faces in.</param>
    /// <param name="blurSigmaFactor">Factor to determine sigma for the gaussian blur. sigma = max(height, width) / factor.</param>
    /// <returns>The number of faces that have been blurred.</returns>
    public static int BlurFaces(this IFaceDetector detector, Image<Rgb24> input, float blurSigmaFactor = 10f)
    {
        var res = detector.DetectFaces(input);
        var cnt = 0;
        input.Mutate(op =>
        {
            foreach (var fc in res)
            {
                var r = Rectangle.Round(fc.Box);
                r.Intersect(input.Bounds());
                var max = Math.Max(r.Width, r.Height);
                var sigma = Math.Max(max / blurSigmaFactor, blurSigmaFactor);
                op.GaussianBlur(sigma, r);
                cnt++;
            }
        });
        return cnt;
    }

    public static void CropProfilePicture(this IFaceDetector detector, Image<Rgb24> input, int? maxEdgeSize = 640, float scaleFactor = 1.35f)
    {
        var res = detector.DetectFaces(input);
        if (res.Count == 0)
        {
            throw new ArgumentException("No faces could be found in the given image", nameof(input));
        }

        var maxFace = res.MaxBy(x => x.Confidence);
        var r = Rectangle.Round(maxFace.Box);
        r = r.ScaleCentered(scaleFactor);
        r.Intersect(input.Bounds());
        var angl = 0.0f;
        if (maxFace.Landmarks is not null && detector is IFaceDetectorWithLandmarks lmdet)
        {
            var (leye, reye) = (lmdet.GetLeftEyeCenter(maxFace.Landmarks), lmdet.GetRightEyeCenter(maxFace.Landmarks));
            angl = leye.GetAlignmentAngle(reye);
        }

        input.Mutate(op =>
        {
            op.CropAligned(r, angl, maxEdgeSize);
        });
    }

    /// <summary>
    /// Counts the number of faces in the given image.
    /// </summary>
    /// <param name="detector">The detector used to search for faces.</param>
    /// <param name="input">The input image.</param>
    /// <returns>The number of faces found.</returns>
    public static int CountFaces(this IFaceDetector detector, Image<Rgb24> input)
    {
        var res = detector.DetectFaces(input);
        return res.Count;
    }

    /// <summary>
    /// Counts the number of faces as well as opened and closed eyes in the given image.
    /// Note that the eye state will only be estimated for eyes with an edge length >= 16.
    /// Thus it is possible that Faces > OpenEyes + ClosedEyes.
    /// </summary>
    /// <param name="detector">The detector used to search for faces.</param>
    /// <param name="eyeStateDetector">The eye state detector.</param>
    /// <param name="input">The input image.</param>
    /// <param name="eyeDistanceDivisor">
    ///     The higher the value given is, the smaller the boxes around the eyes will be.
    ///     Square eye box length = distance between eyes / <paramref name="eyeDistanceDivisor"/> * 2.
    ///     3.0 seems to be a suitable value for typical eye sizes and some additional space around the eyes.
    /// </param>
    /// <returns>The numbers of faces, open and closed eyes found.</returns>
    /// <exception cref="InvalidOperationException">
    ///     Thrown if the face detections by <paramref name="detector"/>
    ///     do not include facial landmarks.
    /// </exception>
    public static (int Faces, int OpenEyes, int ClosedEyes) CountEyeStates(this IFaceDetectorWithLandmarks detector, IEyeStateDetector eyeStateDetector, Image<Rgb24> input, float eyeDistanceDivisor = 3)
    {
        var dets = detector.DetectFaces(input);
        var open = 0;
        var closed = 0;
        foreach (var det in dets)
        {
            var lmrks = det.Landmarks
                ?? throw new InvalidOperationException("Facial landmarks are required but not given for all faces found.");
            var leye = detector.GetLeftEyeCenter(lmrks);
            var reye = detector.GetRightEyeCenter(lmrks);
            var angle = leye.GetAlignmentAngle(reye);
            var bx = ImageCalculations.GetEyeBoxesFromCenterPoints(leye, reye, eyeDistanceDivisor);

            if (Math.Min(bx.Left.Width, bx.Right.Width) < 16)
            {
                continue;
            }

            using var leyeImg = input.CropAligned(bx.Left, angle, 32);
            using var reyeImg = input.CropAligned(bx.Right, angle, 32);
            var leftOpen = eyeStateDetector.IsOpen(leyeImg);
            var rightOpen = eyeStateDetector.IsOpen(reyeImg);
            open += leftOpen ? 1 : 0;
            closed += !leftOpen ? 1 : 0;
            open += rightOpen ? 1 : 0;
            closed += !rightOpen ? 1 : 0;
        }

        return (dets.Count, open, closed);
    }
}
