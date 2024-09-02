// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using BenchmarkDotNet.Attributes;
using FaceAiSharp;
using FaceAiSharp.Extensions;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Benchmarks;

[MemoryDiagnoser]
public class ImageCloneVsMutate
{
    private readonly Image<Rgb24> _img = Image.Load<Rgb24>(@"TestData/jpgs/group_10mpx.jpg");
    private readonly IFaceDetectorWithLandmarks _det;
    private readonly IReadOnlyCollection<FaceDetectorResult> _detectorResults;

    public ImageCloneVsMutate()
    {
        _det = FaceAiSharpBundleFactory.CreateFaceDetectorWithLandmarks();
        _detectorResults = _det.DetectFaces(_img);
    }

    [Benchmark]
    public void Cloning()
    {
        foreach (var res in _detectorResults)
        {
            AlignFaceUsingLandmarksCloning(_img, res.Landmarks!);
        }
    }

    [Benchmark]
    public void Mutating()
    {
        foreach (var res in _detectorResults)
        {
            using var img = _img.Clone();
            AlignFaceUsingLandmarksMutating(img, res.Landmarks!);
        }
    }

    private static Image<Rgb24> AlignFaceUsingLandmarksCloning(Image<Rgb24> face, IReadOnlyList<PointF> landmarks, int edgeSize = 112)
    {
        var cutRect = new Rectangle(0, 0, edgeSize, edgeSize);
        var m = ArcFaceEmbeddingsGenerator.EstimateAffineAlignmentMatrix(landmarks);
        var scaleTo112 = 112f / edgeSize;
        m = Matrix3x2.Multiply(m, Matrix3x2.CreateScale(1 / scaleTo112, 1 / scaleTo112));
        var success = Matrix3x2.Invert(m, out var mi);
        if (!success)
        {
            throw new InvalidOperationException("Could not invert matrix.");
        }

        /* The matrix m transforms the given image in a way that the given landmark points will
         * be projected inside the 112x112 rectangle that is used as input for ArcFace. If the input
         * image is much larger than the face area we are interested in, applying this transform to
         * the complete image would waste cpu time. Thus we first invert the matrix, project our
         * 112x112 crop area using the matrix' inverse and take the minimum surrounding rectangle
         * of that projection. We crop the image using that rectangle and proceed. */
        var area = cutRect.SupersetAreaOfTransform(mi);

        /* The matrix m includes scaling. If we scale the image using an affine transform,
         * we loose quality because we don't use any specialized resizing methods. Thus, we extract
         * the x and y scale factors from the matrix, scale using Resize first and remove the scaling
         * from m by multiplying it with an inverted scale matrix. */
        var (hScale, vScale) = (m.GetHScaleFactor(), m.GetVScaleFactor());
        var mScale = Matrix3x2.CreateScale(1 / hScale, 1 / vScale);
        return face.Clone(op =>
        {
            SafeCrop(op, area);

            var afb = new AffineTransformBuilder();
            var sz = op.GetCurrentSize();
            var scale = new SizeF(sz.Width * hScale, sz.Height * vScale);
            op.Resize(Size.Round(scale));
            m = Matrix3x2.Multiply(mScale, m);

            // the Crop does the inverse translation so we need to undo it
            afb.AppendTranslation(new PointF(Math.Max(0, area.X * hScale), Math.Max(0, area.Y * vScale)));
            afb.AppendMatrix(m);
            op.Transform(afb);

            SafeCrop(op, cutRect);
        });
    }

    private static void AlignFaceUsingLandmarksMutating(Image<Rgb24> face, IReadOnlyList<PointF> landmarks, int edgeSize = 112)
    {
        var cutRect = new Rectangle(0, 0, edgeSize, edgeSize);
        var m = ArcFaceEmbeddingsGenerator.EstimateAffineAlignmentMatrix(landmarks);
        var scaleTo112 = 112f / edgeSize;
        m = Matrix3x2.Multiply(m, Matrix3x2.CreateScale(1 / scaleTo112, 1 / scaleTo112));
        var success = Matrix3x2.Invert(m, out var mi);
        if (!success)
        {
            throw new InvalidOperationException("Could not invert matrix.");
        }

        /* The matrix m transforms the given image in a way that the given landmark points will
         * be projected inside the 112x112 rectangle that is used as input for ArcFace. If the input
         * image is much larger than the face area we are interested in, applying this transform to
         * the complete image would waste cpu time. Thus we first invert the matrix, project our
         * 112x112 crop area using the matrix' inverse and take the minimum surrounding rectangle
         * of that projection. We crop the image using that rectangle and proceed. */
        var area = cutRect.SupersetAreaOfTransform(mi);

        /* The matrix m includes scaling. If we scale the image using an affine transform,
         * we loose quality because we don't use any specialized resizing methods. Thus, we extract
         * the x and y scale factors from the matrix, scale using Resize first and remove the scaling
         * from m by multiplying it with an inverted scale matrix. */
        var (hScale, vScale) = (m.GetHScaleFactor(), m.GetVScaleFactor());
        var mScale = Matrix3x2.CreateScale(1 / hScale, 1 / vScale);
        face.Mutate(op =>
        {
            SafeCrop(op, area);

            var afb = new AffineTransformBuilder();
            var sz = op.GetCurrentSize();
            var scale = new SizeF(sz.Width * hScale, sz.Height * vScale);
            op.Resize(Size.Round(scale));
            m = Matrix3x2.Multiply(mScale, m);

            // the Crop does the inverse translation so we need to undo it
            afb.AppendTranslation(new PointF(Math.Max(0, area.X * hScale), Math.Max(0, area.Y * vScale)));
            afb.AppendMatrix(m);
            op.Transform(afb);

            SafeCrop(op, cutRect);
        });
    }

    private static void SafeCrop(IImageProcessingContext op, Rectangle rect)
    {
        var sz = op.GetCurrentSize();
        var max = new Rectangle(0, 0, sz.Width, sz.Height);
        max.Intersect(rect);
        op.Crop(max);
    }
}
