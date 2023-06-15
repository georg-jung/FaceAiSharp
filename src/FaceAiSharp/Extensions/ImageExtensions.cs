// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using CommunityToolkit.Diagnostics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace FaceAiSharp.Extensions;

public static class ImageExtensions
{
    public static Image<TPixel> CropAligned<TPixel>(this Image<TPixel> sourceImage, Rectangle faceArea, float angle, int? alignedMaxEdgeSize = 250)
        where TPixel : unmanaged, IPixel<TPixel>
        => sourceImage.Clone(op => op.CropAligned(faceArea, angle, alignedMaxEdgeSize));

    public static void CropAlignedDestructive<TPixel>(this Image<TPixel> sourceImage, Rectangle faceArea, float angle, int? alignedMaxEdgeSize = 250)
        where TPixel : unmanaged, IPixel<TPixel>
        => sourceImage.Mutate(op => op.CropAligned(faceArea, angle, alignedMaxEdgeSize));

    internal static void CropAligned(this IImageProcessingContext ctx, Rectangle faceArea, float angle, int? alignedMaxEdgeSize = 250)
    {
        var center = RectangleF.Center(faceArea);
        var minSuperSquare = faceArea.GetMinimumSupersetSquare();

        var angleInv = minSuperSquare.ScaleToRotationAngleInvariantCropArea();
        var bounds = new Rectangle(Point.Empty, ctx.GetCurrentSize());
        if (bounds.Contains(angleInv))
        {
            // If the faceArea we are interested in is at least somewhat smaller than the overall image
            // we can reduce the processing time of CropAligned by multiple orders of magnitude(!) if we
            // crop here first.
            ctx.Crop(angleInv);
            var newBase = new Point(angleInv.X, angleInv.Y);
            var offset = -newBase;
            center.Offset(offset);
            minSuperSquare.Offset(offset);
            faceArea.Offset(offset);
        }

        /* We have cropped off any areas of the image that are completely out of scoped here.
           E.g. if the image is a photo of a group and we have a rough cut of just the one face
           are interested in and removed the other people. */

        if (alignedMaxEdgeSize.HasValue)
        {
            /* We rotate the image below. If we are not interested in a full resolution
             * version of the area for further processing, we can reduce the processing time
             * of CropAligned by a large amount if we resize before rotating. Thus,
             * we scale here by a factor that leaves the final faceArea with exactly the
             * edge size we want to return later. */

            var longestDim = Math.Max(faceArea.Width, faceArea.Height);
            var toLargeFactor = Math.Max(1.0, longestDim / (double)alignedMaxEdgeSize);
            var factor = 1.0 / toLargeFactor; // scale factor

            if (factor < 1)
            {
                var curSize = ctx.GetCurrentSize();
                ctx.Resize(curSize.Scale(factor));

                minSuperSquare = minSuperSquare.Scale(factor);
                faceArea = faceArea.Scale(factor);
                center = RectangleF.Center(faceArea);
            }
        }

        var atb = new AffineTransformBuilder();
        atb.AppendRotationDegrees(angle, center);
        atb.AppendTranslation(new PointF(-minSuperSquare.X, -minSuperSquare.Y));
        ctx.Transform(atb);

        var squareEdge = minSuperSquare.Height;
        var cropArea = new Rectangle(Point.Empty, ctx.GetCurrentSize());
        cropArea.Intersect(new Rectangle(0, 0, squareEdge, squareEdge));
        ctx.Crop(cropArea);

        if (cropArea != minSuperSquare)
        {
            ctx.Resize(new ResizeOptions()
            {
                Position = AnchorPositionMode.TopLeft,
                Mode = ResizeMode.BoxPad,
                PadColor = Color.Black,
                Size = new Size(squareEdge),
            });
        }
    }

    /// <summary>
    /// Returns an image that matches a defined size and PixelFormat. If the given image already conforms to this specification,
    /// it is returned directly. If a conversion is required the pixels of the input image will only be copied exactly once.
    /// If a copy is created, the <see cref="IDisposable"/> value returned equals the <see cref="Image{TPixel}"/> value in the
    /// same tuple. If the passed-in <see cref="Image"/> is returned directly, the <see cref="IDisposable"/> value returned
    /// is null. Thus, you should always use the <see cref="IDisposable"/> in a <c>using</c> block or
    /// <c>using var</c> declaration.
    /// </summary>
    /// <example>
    /// <code>
    /// (var img, var disp) = image.GetProperlySized&lt;Rgb24&gt;(resizeOptions);
    /// using var usingDisp = disp;
    /// </code>
    /// </example>
    /// <typeparam name="TPixel">The pixel format the returned image should have.</typeparam>
    /// <param name="img">The image to return in a proper shape.</param>
    /// <param name="resizeOptions">How to resize the input, if required.</param>
    /// <param name="throwIfResizeRequired">If an actual Resize operation is required to match the spec, throw.</param>
    /// <returns>An <see cref="Image{TPixel}"/> instance sticking to the spec.</returns>
    internal static (Image<TPixel> Image, IDisposable? ToDispose) EnsureProperlySized<TPixel>(this Image img, ResizeOptions resizeOptions, bool throwIfResizeRequired)
        where TPixel : unmanaged, IPixel<TPixel>
    {
        static (Image<TPixel> Image, IDisposable? ToDispose) CreateDisposableTuple(Image<TPixel> img) => (img, img);

        void PerformResize(IImageProcessingContext op) => op.Resize(resizeOptions);

        Image<TPixel> CreateProperSizedImageSameFormat(Image<TPixel> img) => img.Clone(PerformResize);

        Image<TPixel> CreateProperSizedImage(Image img)
        {
            var ret = img.CloneAs<TPixel>();
            ret.Mutate(PerformResize);
            return ret;
        }

        var (wR, hR) = (resizeOptions.Size.Width, resizeOptions.Size.Height); // r = required
        var (wA, hA) = (img.Width, img.Height); // a = actual
        return img switch
        {
            Image<TPixel> rgbImg when wA == wR && hA == hR => (rgbImg, null),
            Image<TPixel> rgbImg when !throwIfResizeRequired => CreateDisposableTuple(CreateProperSizedImageSameFormat(rgbImg)),
            Image when wA == wR && hA == hR => CreateDisposableTuple(img.CloneAs<TPixel>()),
            Image when !throwIfResizeRequired => CreateDisposableTuple(CreateProperSizedImage(img)),
            _ => throw new ArgumentException($"The given image does not have the required dimensions (Required: W={wR}, H={hR}; Actual: W={wA}, H={hA})"),
        };
    }

    internal static void EnsureProperlySizedDestructive(this Image<Rgb24> img, ResizeOptions resizeOptions, bool throwIfResizeRequired)
    {
        void PerformResize(IImageProcessingContext op) => op.Resize(resizeOptions);

        var req = (resizeOptions.Size.Width, resizeOptions.Size.Height);
        var act = (img.Width, img.Height);

        if (req == act)
        {
            return;
        }

        if (throwIfResizeRequired)
        {
            Guard.IsEqualTo(req, act);
        }

        img.Mutate(PerformResize);
    }
}
