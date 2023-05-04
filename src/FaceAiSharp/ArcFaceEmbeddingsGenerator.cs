// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;
using CommunityToolkit.Diagnostics;
using FaceAiSharp.Abstractions;
using FaceAiSharp.Extensions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace FaceAiSharp;

public sealed class ArcFaceEmbeddingsGenerator : IFaceEmbeddingsGenerator, IDisposable
{
    /*  --- FINISHED! Final Stats 0,275 ---
        Calculated 6000 pairs.
        Had 3000 pairs belonging to the same person.
        Avergage cosine distance:                   0,6669038685361545
        Avergage euclidean distance:                1,1050784109483163
        Avergage dot product:                       0,3330961314615561
        Avergage cosine distance [same person]:     0,34073335826396944
        Avergage euclidean distance [same person]:  0,8019862072567145
        Avergage dot product [same person]:         0,6592666412101438
        Avergage cosine distance [diff. person]:    0,9930743788083395
        Avergage euclidean distance [diff. person]: 1,408170614639918
        Avergage dot product [diff. person]:        0,006925621712968374
        P:     3000 TP:     2822 FP:        1
        N:     3000 TN:     2999 FN:      178
        Accuracy:  97,02 %
        Precision: 99,96 %
        Recall:    94,07 %
        F1 score:  0,9693
        AuROC:     0,9739247
        Threshold for best accuracy: 0,28119734
    */

    /// <summary>
    /// Points from https://github.com/deepinsight/insightface/blob/c7bf2048e8947a6398b4b8bda6d1958138fdc9b5/python-package/insightface/utils/face_align.py.
    /// </summary>
    private static readonly IReadOnlyList<PointF> ExpectedLandmarkPositionsInsightface = new List<PointF>()
    {
        new PointF(38.2946f, 51.6963f),
        new PointF(73.5318f, 51.5014f),
        new PointF(56.0252f, 71.7366f),
        new PointF(41.5493f, 92.3655f),
        new PointF(70.7299f, 92.2041f),
    }.AsReadOnly();

    /*  --- FINISHED! Final Stats 0,264 ---
        Calculated 6000 pairs.
        Had 3000 pairs belonging to the same person.
        Avergage cosine distance:                   0,6998155847887199
        Avergage euclidean distance:                1,149087769721945
        Avergage dot product:                       0,30018441526552975
        Avergage cosine distance [same person]:     0,4252110035220782
        Avergage euclidean distance [same person]:  0,9035951985418796
        Avergage dot product [same person]:         0,5747889964582088
        Avergage cosine distance [diff. person]:    0,9744201660553614
        Avergage euclidean distance [diff. person]: 1,3945803409020106
        Avergage dot product [diff. person]:        0,025579834072850645
        P:     3000 TP:     2804 FP:       13
        N:     3000 TN:     2987 FN:      196
        Accuracy:  96,52 %
        Precision: 99,54 %
        Recall:    93,47 %
        F1 score:  0,9641
        AuROC:     0,97111547
        Threshold for best accuracy: 0,26378733
    */

    /// <summary>
    /// Points from https://github.com/onnx/models/blob/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/body_analysis/arcface/dependencies/arcface_inference.ipynb.
    /// </summary>
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Major Code Smell", "S1144:Unused private types or members should be removed", Justification = "We want to keep these for possible future use")]
    private static readonly IReadOnlyList<PointF> ExpectedLandmarkPositionsOnnxZoo = new List<PointF>()
    {
        new PointF(30.2946f, 51.6963f),
        new PointF(65.5318f, 51.5014f),
        new PointF(48.0252f, 71.7366f),
        new PointF(33.5493f, 92.3655f),
        new PointF(62.7299f, 92.2041f),
    }.AsReadOnly();

    private static readonly ResizeOptions _resizeOptions = new()
    {
        Mode = ResizeMode.Pad,
        PadColor = Color.Black,
        Size = new Size(112, 112),
    };

    private readonly InferenceSession _session;

    /// <summary>
    /// Initializes a new instance of the <see cref="ArcFaceEmbeddingsGenerator"/> class.
    /// </summary>
    /// <param name="options">Provide a path to the ONNX model file and customize the behaviour of <see cref="ArcFaceEmbeddingsGenerator"/>.</param>
    /// <param name="sessionOptions"><see cref="SessionOptions"/> to customize OnnxRuntime's behaviour.</param>
    public ArcFaceEmbeddingsGenerator(ArcFaceEmbeddingsGeneratorOptions options, SessionOptions? sessionOptions = null)
    {
        _ = options?.ModelPath ?? throw new ArgumentException("A model path is required in options.ModelPath.", nameof(options));
        Options = options;
        _session = sessionOptions is null ? new(options.ModelPath) : new(options.ModelPath, sessionOptions);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ArcFaceEmbeddingsGenerator"/> class.
    /// </summary>
    /// <param name="model">An ONNX model containing the ResNet100 model with 1x3x112x112 input dimensions.</param>
    /// <param name="options">Options to customize the behaviour of <see cref="ArcFaceEmbeddingsGenerator"/>. If options.ModelPath is set, it is ignored. The model provided in <paramref name="model"/> takes precedence.</param>
    /// <param name="sessionOptions"><see cref="SessionOptions"/> to customize OnnxRuntime's behaviour.</param>
    public ArcFaceEmbeddingsGenerator(byte[] model, ArcFaceEmbeddingsGeneratorOptions? options = null, SessionOptions? sessionOptions = null)
    {
        _ = model ?? throw new ArgumentNullException(nameof(model));
        Options = options ?? new();
        _session = sessionOptions is null ? new(model) : new(model, sessionOptions);
    }

    public ArcFaceEmbeddingsGeneratorOptions Options { get; }

    /// <summary>
    /// Transform and crop the given image in the way ArcFace was trained.
    /// </summary>
    /// <param name="face">Image containing the face. The given image will be mutated.</param>
    /// <param name="landmarks">5 facial landmark points.</param>
    /// <param name="edgeSize">
    ///     ArcFace typically expects 112x112 inputs. Specifying another edge size might be useful for
    ///     custom models or to manually analyze the resulting aligned faces.</param>
    public static void AlignUsingFacialLandmarks(Image face, IReadOnlyList<PointF> landmarks, int edgeSize = 112)
    {
        var cutRect = new Rectangle(0, 0, edgeSize, edgeSize);
        var m = EstimateAffineAlignmentMatrix(landmarks);
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

    public void Dispose() => _session.Dispose();

    public float[] Generate(Image<Rgb24> alignedFace)
    {
        alignedFace.EnsureProperlySizedDestructive(_resizeOptions, !Options.AutoResizeInputToModelDimensions);

        var input = CreateImageTensor(alignedFace);

        var inputMeta = _session.InputMetadata;
        var name = inputMeta.Keys.First();

        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(name, input) };
        using var outputs = _session.Run(inputs);
        var firstOut = outputs.First();
        var tens = firstOut.Value as DenseTensor<float> ?? firstOut.AsTensor<float>().ToDenseTensor();
        Debug.Assert(tens.Length % 512 == 0, "Output tensor length is invalid.");

        var embSpan = tens.Buffer.Span;
        return GeometryExtensions.ToUnitLength(embSpan);
    }

    internal static DenseTensor<float> CreateImageTensor(Image<Rgb24> img)
    {
        if (img.Height != 112 || img.Width != 112)
        {
            throw new ArgumentException("The given image must be 112x112 pixels.", nameof(img));
        }

        // ArcFace uses the rgb values directly, just the ints converted to float,
        // no further preprocessing needed. The default ToTensor implementation assumes
        // we want the RGB[
        var mean = new[] { 0f, 0f, 0f };
        var stdDevVal = 1 / 255f;
        var stdDev = new[] { stdDevVal, stdDevVal, stdDevVal };
        var inputDim = new[] { 1, 3, 112, 112 };
        return img.ToTensor(mean, stdDev, inputDim);
    }

    internal static System.Numerics.Matrix3x2 EstimateAffineAlignmentMatrix(IReadOnlyList<PointF> landmarks)
    {
        Guard.HasSizeEqualTo(landmarks, 5);
        var estimate = new List<(PointF A, PointF B)>
        {
            (landmarks[0], ExpectedLandmarkPositionsInsightface[0]),
            (landmarks[1], ExpectedLandmarkPositionsInsightface[1]),
            (landmarks[2], ExpectedLandmarkPositionsInsightface[2]),
            (landmarks[3], ExpectedLandmarkPositionsInsightface[3]),
            (landmarks[4], ExpectedLandmarkPositionsInsightface[4]),
        };
        var m = estimate.EstimateSimilarityMatrix();
        return m;
    }

    private static void SafeCrop(IImageProcessingContext op, Rectangle rect)
    {
        var sz = op.GetCurrentSize();
        var max = new Rectangle(0, 0, sz.Width, sz.Height);
        max.Intersect(rect);
        op.Crop(max);
    }
}

public record ArcFaceEmbeddingsGeneratorOptions
{
    /// <summary>
    /// Gets the path to the ONNX file that contains the ResNet100 model with 1x3x112x112 input dimensions.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Resize the image to dimensions supported by the model if required. This detector throws an
    /// exception if this is set to false and an image is passed in unsupported dimensions.
    /// </summary>
    public bool AutoResizeInputToModelDimensions { get; set; } = true;
}
