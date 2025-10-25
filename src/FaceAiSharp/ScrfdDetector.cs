// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using FaceAiSharp.Extensions;
using FaceAiSharp.Simd;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using SimpleSimd;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace FaceAiSharp;

public sealed class ScrfdDetector : IFaceDetectorWithLandmarks, IDisposable
{
    private readonly InferenceSession _session;
    private readonly ModelParameters _modelParameters;
    private readonly IMemoryCache _cache;

    /// <summary>
    /// Initializes a new instance of the <see cref="ScrfdDetector"/> class.
    /// </summary>
    /// <param name="cache">An <see cref="IMemoryCache"/> instance that is used internally by <see cref="ScrfdDetector"/>.</param>
    /// <param name="options">Provide a path to the ONNX model file and customize the behaviour of <see cref="ScrfdDetector"/>.</param>
    /// <param name="sessionOptions"><see cref="SessionOptions"/> to customize OnnxRuntime's behaviour.</param>
    public ScrfdDetector(IMemoryCache cache, ScrfdDetectorOptions options, SessionOptions? sessionOptions = null)
    {
        _ = options?.ModelPath ?? throw new ArgumentException("A model path is required in options.ModelPath.", nameof(options));
        Options = options;
        _cache = cache;
        _session = sessionOptions is null ? new(options.ModelPath) : new(options.ModelPath, sessionOptions);

        _modelParameters = DetermineModelParameters();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ScrfdDetector"/> class.
    /// </summary>
    /// <param name="model">An scrfd onnx model that supports facial landmarks ("kps").</param>
    /// <param name="cache">An <see cref="IMemoryCache"/> instance that is used internally by <see cref="ScrfdDetector"/>.</param>
    /// <param name="options">Options to customize the behaviour of <see cref="ScrfdDetector"/>. If options.ModelPath is set, it is ignored. The model provided in <paramref name="model"/> takes precedence.</param>
    /// <param name="sessionOptions"><see cref="SessionOptions"/> to customize OnnxRuntime's behaviour.</param>
    public ScrfdDetector(byte[] model, IMemoryCache cache, ScrfdDetectorOptions? options = null, SessionOptions? sessionOptions = null)
    {
        _ = model ?? throw new ArgumentNullException(nameof(model));
        Options = options ?? new();
        _cache = cache;
        _session = sessionOptions is null ? new(model) : new(model, sessionOptions);

        _modelParameters = DetermineModelParameters();
    }

    public ScrfdDetectorOptions Options { get; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static PointF GetLeftEye(IReadOnlyList<PointF> landmarks) => landmarks[0];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static PointF GetRightEye(IReadOnlyList<PointF> landmarks) => landmarks[1];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static PointF GetNose(IReadOnlyList<PointF> landmarks) => landmarks[2];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static PointF GetMouthLeft(IReadOnlyList<PointF> landmarks) => landmarks[3];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static PointF GetMouthRight(IReadOnlyList<PointF> landmarks) => landmarks[4];

    public IReadOnlyCollection<FaceDetectorResult> DetectFaces(Image<Rgb24> image)
    {
        var targetSize = _modelParameters.InputSize ?? new Size((int)Math.Ceiling(image.Width / 32.0) * 32, (int)Math.Ceiling(image.Height / 32.0) * 32);
        if (Options.MaximumInputSize is Size maxSz && (targetSize.Width > maxSz.Width || targetSize.Height > maxSz.Height))
        {
            targetSize = maxSz;
        }

        var resizeOptions = new ResizeOptions()
        {
            Size = targetSize,
            Position = AnchorPositionMode.TopLeft,
            Mode = ResizeMode.BoxPad,
            PadColor = Color.Black,
        };

        (var img, var disp) = image.EnsureProperlySized<Rgb24>(resizeOptions, !Options.AutoResizeInputToModelDimensions);
        using var usingDisp = disp;
        var scale = 1 / image.Bounds.GetScaleFactorToFitInto(resizeOptions.Size);

        var input = CreateImageTensor(img);
        return Detect(input, img.Size, scale);
    }

    IReadOnlyList<PointF> IFaceLandmarksDetector.DetectLandmarks(Image<Rgb24> image) => DetectFaces(image).MaxBy(x => x.Confidence).Landmarks!;

    PointF IFaceLandmarksDetector.GetLeftEyeCenter(IReadOnlyList<PointF> landmarks) => GetLeftEye(landmarks);

    PointF IFaceLandmarksDetector.GetRightEyeCenter(IReadOnlyList<PointF> landmarks) => GetRightEye(landmarks);

    public void Dispose() => _session.Dispose();

    internal static DenseTensor<float> CreateImageTensor(Image<Rgb24> img) => img.ToTensor();

    /// <summary>
    /// Filter out duplicate detections (multiple boxes describing roughly the same area) using non max suppression.
    /// </summary>
    /// <param name="dets">All detections with their scores.</param>
    /// <param name="thresh">Non max suppression threshold.</param>
    /// <returns>Which detections to keep.</returns>
    internal static List<int> NonMaxSupression(IReadOnlyList<FaceDetectorResult> orderedResults, float thresh)
    {
        float[] x1 = [.. orderedResults.Select(x => x.Box.Left)];
        float[] x2 = [.. orderedResults.Select(x => x.Box.Right)];
        float[] y1 = [.. orderedResults.Select(x => x.Box.Top)];
        float[] y2 = [.. orderedResults.Select(x => x.Box.Bottom)];

        int len = x1.Length;
        var p = SimdOps<float>.Subtract(x2, x1);
        var q = SimdOps<float>.Subtract(y2, y1);
        SimdOps<float>.Add(p, 1f, p);
        SimdOps<float>.Add(q, 1f, q);
        var areasArr = SimdOps<float>.Multiply(p, q);
        var areas = areasArr.AsSpan();

        var discard = new int[len];
        var pivot_discard = new int[len];
        var xx1 = new float[len];
        var xx2 = new float[len];
        var yy1 = new float[len];
        var yy2 = new float[len];
        var keep = new List<int>(len);
        for (var i = 0; i < len; i++)
        {
            if (discard[i] != 0)
            {
                continue;
            }

            keep.Add(i);

            var (i_x1, i_x2, i_y1, i_y2, i_area) = (x1[i], x2[i], y1[i], y2[i], areas[i]);
            ElementwiseMax.Max(x1, i_x1, xx1);
            ElementwiseMax.Max(y1, i_y1, yy1);
            ElementwiseMin.Min(x2, i_x2, xx2);
            ElementwiseMin.Min(y2, i_y2, yy2);

            SimdOps<float>.Subtract(xx2, xx1, xx1);
            SimdOps<float>.Add(xx1, 1f, xx1);
            ElementwiseMax.Max(xx1, 0f, xx1); // xx1 = w

            SimdOps<float>.Subtract(yy2, yy1, yy1);
            SimdOps<float>.Add(yy1, 1f, yy1);
            ElementwiseMax.Max(yy1, 0f, yy1); // yy1 = h

            SimdOps<float>.Multiply(xx1, yy1, xx1); // xx1 = inter
            SimdOps<float>.Add(areas, i_area, yy1);
            SimdOps<float>.Subtract(yy1, xx1, yy1); // yy1 = (areas[i] + areas[order["1:"]] - inter)
            SimdOps<float>.Divide(xx1, yy1, xx1); // xx1 = ovr = inter / (areas[i] + areas[order["1:"]] - inter)

            ElementwiseGreater.Greater(xx1, thresh, pivot_discard);
            SimdOps<int>.Add(discard, pivot_discard, discard);
        }

        return keep;
    }

    internal IReadOnlyCollection<FaceDetectorResult> Detect(DenseTensor<float> input, Size imgSize, float scale)
    {
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_modelParameters.InputName, input) };
        using var outputs = _session.Run(inputs);

        var strideResults = new List<FaceDetectorResult>();
        foreach (var (idx, stride) in _modelParameters.FeatStrideFpn.Select((val, idx) => (idx, val)))
        {
            var strideRes = HandleStride(idx, stride, outputs.ToList(), imgSize, _modelParameters.Batching);
            strideResults.AddRange(strideRes ?? []);
        }

        if (strideResults.Count == 0)
        {
            return [];
        }

        strideResults.Sort((a, b) => b.Confidence!.Value.CompareTo(a.Confidence!.Value));
        var keepIdxs = NonMaxSupression(strideResults, Options.NonMaxSupressionThreshold);
        return [.. keepIdxs.Select(i => strideResults[i])];
    }

    internal static (int X, int Y) GetAnchorCenter(Size inputSize, int stride, int numAnchors, int anchorIdx)
    {
        var height = inputSize.Height / stride;
        var width = inputSize.Width / stride;
        var anchorsPerRow = width * numAnchors;
        var y = anchorIdx / anchorsPerRow;
        var x = (anchorIdx % anchorsPerRow) / numAnchors;

        return (x * stride, y * stride);
    }

    internal static float[] GenerateAnchorCenters(Size inputSize, int stride, int numAnchors)
    {
        // see also https://github.com/deepinsight/insightface/blob/f091989568cad5a0244e05be1b8d58723de210b0/detection/scrfd/tools/scrfd.py#L185
        var height = inputSize.Height / stride;
        var width = inputSize.Width / stride;
        var data = new float[height * width * numAnchors * 2];
        var idx = 0;

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                for (var a = 0; a < numAnchors; a++)
                {
                    data[idx++] = x * stride;
                    data[idx++] = y * stride;
                }
            }
        }

        return data;
    }

    /// <summary>
    /// Gets the indices of elements that are larger than or equal to a threshold.
    /// </summary>
    /// <param name="input">The array to search.</param>
    /// <param name="threshold">The threshold value. Inclusive.</param>
    /// <returns>An array of indices where input[i] >= threshold.</returns>
    private static List<int> IndicesOfElementsLargerThanOrEqual(float[] input, float threshold)
    {
        var indices = new List<int>();
        for (var i = 0; i < input.Length; i++)
        {
            if (input[i] >= threshold)
            {
                indices.Add(i);
            }
        }

        return indices;
    }

    private static NDArray Distance2Bbox(NDArray points, NDArray distance)
    {
        var x1 = points[":, 0"] - distance[":, 0"];
        var y1 = points[":, 1"] - distance[":, 1"];
        var x2 = points[":, 0"] + distance[":, 2"];
        var y2 = points[":, 1"] + distance[":, 3"];
        return np.stack(new[] { x1, y1, x2, y2 }, axis: -1);
    }

    private static NDArray Distance2Kps(NDArray points, NDArray distance)
    {
        var preds = new NDArray[distance.shape[1]];
        for (var i = 0; i < distance.shape[1]; i += 2)
        {
            var px = points[Slice.All, i % 2] + distance[Slice.All, i];
            var py = points[Slice.All, (i % 2) + 1] + distance[Slice.All, i + 1];
            preds[i] = px;
            preds[i + 1] = py;
        }

        return np.stack(preds, axis: -1);
    }

    private static IReadOnlyList<PointF> Kps(ReadOnlySpan<float> flatKps, int anchorX, int anchorY, int stride)
    {
        return [
            new(anchorX + (flatKps[0] * stride), anchorY + (flatKps[1] * stride)),
            new(anchorX + (flatKps[2] * stride), anchorY + (flatKps[3] * stride)),
            new(anchorX + (flatKps[4] * stride), anchorY + (flatKps[5] * stride)),
            new(anchorX + (flatKps[6] * stride), anchorY + (flatKps[7] * stride)),
            new(anchorX + (flatKps[8] * stride), anchorY + (flatKps[9] * stride)),
        ];
    }

    private List<FaceDetectorResult>? HandleStride(int strideIndex, int stride, IReadOnlyList<NamedOnnxValue> outputs, Size inputSize, bool batched)
    {
        var thresh = Options.ConfidenceThreshold;
        var scores = outputs[strideIndex].ToArray<float>();
        var indicesAboveThreshold = IndicesOfElementsLargerThanOrEqual(scores, thresh);
        if (indicesAboveThreshold.Count == 0)
        {
            return null;
        }

        var bboxPreds = outputs[strideIndex + _modelParameters.Fmc].ToArray<float>();
        var kpsPreds = outputs.ElementAtOrDefault(strideIndex + (_modelParameters.Fmc * 2))?.ToArray<float>();

        var returnValues = new List<FaceDetectorResult>(indicesAboveThreshold.Count);
        foreach (var anchorIdx in indicesAboveThreshold)
        {
            var (x, y) = GetAnchorCenter(inputSize, stride, _modelParameters.NumAnchors, anchorIdx);
            var bboxBaseIdx = anchorIdx * 4;
            var (x0diff, y0diff, x1diff, y1diff) = (bboxPreds[bboxBaseIdx + 0] * stride, bboxPreds[bboxBaseIdx + 1] * stride, bboxPreds[bboxBaseIdx + 2] * stride, bboxPreds[bboxBaseIdx + 3] * stride);
            var bbox = new RectangleF(x - x0diff, y - y0diff, x0diff + x1diff, y0diff + y1diff);
            if (kpsPreds is not null)
            {
                var kpsBaseIdx = anchorIdx * 10;
                var kps = Kps(kpsPreds.AsSpan(kpsBaseIdx, 10), x, y, stride);
                returnValues.Add(new FaceDetectorResult(bbox, kps, scores[anchorIdx]));
            }
            else
            {
                returnValues.Add(new FaceDetectorResult(bbox, null, scores[anchorIdx]));
            }
        }

        return returnValues;
    }

    private NDArray GetAnchorCenters(Size inputSize, int stride, int numAnchors)
    => _cache.GetOrCreate((inputSize, stride, numAnchors), cacheEntry =>
    {
        cacheEntry.SetSlidingExpiration(TimeSpan.FromMinutes(20));
        var floatData = GenerateAnchorCenters(inputSize, stride, numAnchors);
        var nda = new NDArray(floatData, new Shape(floatData.Length / 2, 2));
        return nda;
    })!;

    private ModelParameters DetermineModelParameters()
    {
        var inputMeta = _session.InputMetadata;
        var inputName = inputMeta.Keys.First();
        var input = inputMeta[inputName];
        var (x, y) = (input.Dimensions[2], input.Dimensions[3]);
        var inputSize = x == -1 && y == -1 ? (Size?)null : new Size(x, y);

        var firstOutput = _session.OutputMetadata.Values.First();
        var batched = firstOutput.Dimensions.Length == 3;
        if (batched)
        {
            throw new NotImplementedException("Batched models are not supported at this time.");
        }

        var (fmc, stridesFpn, kps, numAnchors) = _session.OutputMetadata.Values.Count() switch
        {
            6 => (3, new[] { 8, 16, 32 }, false, 2),
            9 => (3, new[] { 8, 16, 32 }, true, 2),
            10 => (5, new[] { 8, 16, 32, 64, 128 }, false, 1),
            15 => (5, new[] { 8, 16, 32, 64, 128 }, true, 1),
            int cnt => throw new NotSupportedException($"{cnt} output tensors are not supported for SCRFD models."),
        };

        return new(inputSize, inputName, batched, fmc, stridesFpn, kps, numAnchors);
    }

    /// <summary>
    /// Loosely on https://github.com/deepinsight/insightface/blob/e8c33fc91f60c28f088415864df9d200e11c4c30/python-package/insightface/model_zoo/scrfd.py#L88.
    /// </summary>
    /// <param name="InputSize">Input dimensions.</param>
    /// <param name="InputName">Name of the input tensor.</param>
    /// <param name="Batching">True if the model supports batching.</param>
    /// <param name="Fmc">Probably "Feature Map Count", referring to the number of feature maps produced by the model.</param>
    /// <param name="FeatStrideFpn">Probably "Feature Stride Feature Pyramid Network", referring to the stride of the feature pyramid network.</param>
    /// <param name="SupportsKps">True if the model supports facial landmarks.</param>
    /// <param name="NumAnchors">Number of anchors.</param>
    private readonly record struct ModelParameters(Size? InputSize, string InputName, bool Batching, int Fmc, int[] FeatStrideFpn, bool SupportsKps, int NumAnchors);
}

public record ScrfdDetectorOptions
{
    /// <summary>
    /// Gets the path to the onnx file that contains the scrfd model that supports facial landmarks ("kps").
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Resize the image to dimensions supported by the model if required. This detector throws an
    /// exception if this is set to false and an image is passed in unsupported dimensions.
    /// </summary>
    public bool AutoResizeInputToModelDimensions { get; set; } = true;

    public float NonMaxSupressionThreshold { get; set; } = 0.4f;

    public float ConfidenceThreshold { get; set; } = 0.5f;

    public Size? MaximumInputSize { get; set; } = new(640, 640);
}
