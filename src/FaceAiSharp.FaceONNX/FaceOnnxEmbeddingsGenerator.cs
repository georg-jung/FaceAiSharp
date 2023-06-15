// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using FaceAiSharp.Extensions;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceAiSharp;

public sealed class FaceOnnxEmbeddingsGenerator : IFaceEmbeddingsGenerator, IDisposable
{
    private readonly FaceONNX.FaceEmbedder _fonnx = new();

    public void Dispose() => _fonnx.Dispose();

    public float[] GenerateEmbedding(Image<Rgb24> alignedFace)
    {
        var img = alignedFace.ToFaceOnnxFloatArray();
        var res = _fonnx.Forward(img);
        return res;
    }

    void IFaceEmbeddingsGenerator.AlignFaceUsingLandmarks(Image<Rgb24> face, IReadOnlyList<PointF> landmarks) => throw new NotImplementedException();
}
