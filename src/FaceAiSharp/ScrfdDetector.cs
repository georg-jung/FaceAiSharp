// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using FaceAiSharp.Extensions;
using FaceAiSharp.Simd;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using NumSharp.Backends.Unmanaged;
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
        var scale = 1 / image.Bounds().GetScaleFactorToFitInto(resizeOptions.Size);

        var input = CreateImageTensor(img);
        return Detect(input, img.Size(), scale);
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
    internal static List<int> NonMaxSupression(NDArray dets, float thresh)
    {
        var x1 = ((IArraySlice)dets[":, 0"].Data<float>()).AsSpan<float>();
        var y1 = ((IArraySlice)dets[":, 1"].Data<float>()).AsSpan<float>();
        var x2 = ((IArraySlice)dets[":, 2"].Data<float>()).AsSpan<float>();
        var y2 = ((IArraySlice)dets[":, 3"].Data<float>()).AsSpan<float>();

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

        List<NDArray> scoresLst = new(_modelParameters.Fmc);
        List<NDArray> bboxesLst = new(_modelParameters.Fmc);
        List<NDArray> kpssLst = new(_modelParameters.Fmc);

        foreach (var (idx, stride) in _modelParameters.FeatStrideFpn.Select((val, idx) => (idx, val)))
        {
            var strideRes = HandleStride(idx, stride, outputs.ToList(), imgSize, _modelParameters.Batching);
            if (!strideRes.HasValue)
            {
                continue;
            }

            var (s, bb, kps) = strideRes.Value;
            scoresLst.Add(s);
            bboxesLst.Add(bb);
            if (kps is not null)
            {
                kpssLst.Add(kps);
            }
        }

        var scores = scoresLst.Count != 0 ? np.vstack(scoresLst.ToArray()) : null;
        if (scores is null || scores.size == 0)
        {
            return new List<FaceDetectorResult>(0);
        }

        var scores_ravel = scores.ravel();
        var order = scores_ravel.argsort<float>()["::-1"];
        var bboxes = np.vstack(bboxesLst.ToArray());

        var preDet = np.hstack(bboxes, scores);

        preDet = preDet[order];
        var keep = np.array(NonMaxSupression(preDet, Options.NonMaxSupressionThreshold));
        var det = preDet[keep];
        det *= scale;

        NDArray? kpss = null;
        if (kpssLst.Count > 0)
        {
            kpss = np.vstack(kpssLst.ToArray());
            kpss = kpss[order];
            kpss = kpss[keep];
            kpss *= scale;
        }

        static FaceDetectorResult ToReturnType(NDArray input)
        {
            var x1 = input.GetSingle(0);
            var y1 = input.GetSingle(1);
            var x2 = input.GetSingle(2);
            var y2 = input.GetSingle(3);
            return new(new RectangleF(x1, y1, x2 - x1, y2 - y1), null, input.GetSingle(4));
        }

        static FaceDetectorResult ToReturnTypeWithLandmarks(NDArray input, NDArray kps)
        {
            var (box, _, conf) = ToReturnType(input);
            var lmrks = new List<PointF>(5); // don't use ToList because we know we will always have eactly 5.
            lmrks.AddRange(kps.GetNDArrays(0).Select(x => new PointF(x.GetSingle(0), x.GetSingle(1))));
            return new(box, lmrks, conf);
        }

        if (kpss is not null)
        {
            return det.GetNDArrays(0).Zip(kpss.GetNDArrays(0)).Select(x => ToReturnTypeWithLandmarks(x.First, x.Second)).ToList();
        }
        else
        {
            return det.GetNDArrays(0).Select(ToReturnType).ToList();
        }
    }

    private static NDArray GenerateAnchorCenters(Size inputSize, int stride, int numAnchors)
    {
        // translated from https://github.com/deepinsight/insightface/blob/f091989568cad5a0244e05be1b8d58723de210b0/detection/scrfd/tools/scrfd.py#L185
        var height = inputSize.Height / stride;
        var width = inputSize.Width / stride;
        var (mgrid1, mgrid2) = np.mgrid(np.arange(height), np.arange(width));
        var anchorCenters = np.stack(new[] { mgrid2, mgrid1 }, axis: -1).astype(np.float32);
        anchorCenters = (anchorCenters * stride).reshape(-1, 2);
        if (numAnchors > 1)
        {
            anchorCenters = np.stack(new[] { anchorCenters, anchorCenters }, axis: 1).reshape(-1, 2);
        }

        return anchorCenters;
    }

    /// <summary>
    /// In real numpy this could be eg. np.where(scores&gt;=thresh) or np.asarray(condition).nonzero().
    /// </summary>
    /// <param name="input">The indices of this array's elements should be returned.</param>
    /// <param name="threshold">The threshold value. Exclusive.</param>
    /// <returns>An NDArray contianing the indices.</returns>
    private static NDArray IndicesOfElementsLargerThen(NDArray input, float threshold)
    {
        var zeroIfBelow = np.sign(input - threshold) + 1;
        var ret = np.nonzero(zeroIfBelow);
        return ret[0];
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

    private (NDArray Scores, NDArray Bboxes, NDArray? Kpss)? HandleStride(int strideIndex, int stride, IReadOnlyList<NamedOnnxValue> outputs, Size inputSize, bool batched)
    {
        var thresh = Options.ConfidenceThreshold;
        var scores = outputs[strideIndex].ToNDArray<float>();
        var bbox_preds = outputs[strideIndex + _modelParameters.Fmc].ToNDArray<float>();
        var kps_preds = outputs.ElementAtOrDefault(strideIndex + (_modelParameters.Fmc * 2))?.ToNDArray<float>();

        if (batched)
        {
            bbox_preds = bbox_preds[0];
            scores = scores[0];
            kps_preds = kps_preds?[0];
        }

        bbox_preds *= stride;
        kps_preds = kps_preds is not null ? kps_preds * stride : null;

        var anchorCenters = GetAnchorCenters(inputSize, stride, _modelParameters.NumAnchors);

        // this is >= in python but > here
        var pos_inds = IndicesOfElementsLargerThen(scores, thresh);
        if (pos_inds.size == 0)
        {
            return null;
        }

        anchorCenters = anchorCenters[pos_inds];
        bbox_preds = bbox_preds[pos_inds];
        scores = scores[pos_inds];

        var bboxes = Distance2Bbox(anchorCenters, bbox_preds);
        NDArray? kpss = null;

        if (kps_preds is not null)
        {
            kps_preds = kps_preds[pos_inds];
            kpss = Distance2Kps(anchorCenters, kps_preds);
            kpss = kpss.reshape(kpss.shape[0], -1, 2);
        }

        return (scores, bboxes, kpss);
    }

    private NDArray GetAnchorCenters(Size inputSize, int stride, int numAnchors)
    => _cache.GetOrCreate((inputSize, stride, numAnchors), cacheEntry =>
    {
        cacheEntry.SetSlidingExpiration(TimeSpan.FromMinutes(20));
        return GenerateAnchorCenters(inputSize, stride, numAnchors);
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
