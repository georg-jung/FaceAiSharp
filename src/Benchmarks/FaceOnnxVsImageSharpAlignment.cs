// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Attributes;
using FaceAiSharp.Extensions;
using FaceONNX;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

namespace Benchmarks;

[MemoryDiagnoser]
public class FaceOnnxVsImageSharpAlignment
{
    private readonly Image _img = Image.Load(@"C:\Users\georg\OneDrive\Bilder\20160111-0162_GJ.jpg");
    private readonly float[][,] _asFloat;
    private readonly Rectangle _rect = new(50, 99, 120, 240);
    private readonly System.Drawing.Rectangle _drawingRect = new(50, 99, 120, 240);
    private readonly float _angle = 23.45f;

    public FaceOnnxVsImageSharpAlignment()
    {
        _asFloat = _img.ToFaceOnnxFloatArrayParallel();
    }

    public static Image Align(Image sourceImage, Rectangle faceArea, float angle)
        => sourceImage.Clone(op =>
            {
                var angleInvariantCropArea = faceArea.ScaleToRotationAngleInvariantCropArea();
                op.Crop(faceArea);
                op.Rotate(angle);

                // We have cropped above to an area that is larger than our actual face area.
                // It is exactly so large that it fits every possible rotation of the given face
                // area around any angle, rotated around it's center. Thus, we don't have black/blank
                // areas after applying the rotation. Now, we do want to crop the rotated image to our
                // actual faceArea.
                var cropAreaAfterRotation = new Rectangle()
                {
                    X = faceArea.X - angleInvariantCropArea.X,
                    Y = faceArea.Y - angleInvariantCropArea.Y,
                    Height = faceArea.Height,
                    Width = faceArea.Width,
                };
                op.Crop(cropAreaAfterRotation);
            });

    [Benchmark]
    public void ImageSharpAlign() => Align(_img, _rect, _angle);

    [Benchmark]
    public void FaceOnnxAlign() => FaceLandmarksExtractor.Align(_asFloat, _drawingRect, _angle);
}
