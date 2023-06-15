// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using FaceAiSharp.Extensions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace FaceAiSharp;

public sealed class OpenVinoOpenClosedEye0001 : IEyeStateDetector, IDisposable
{
    private static readonly ResizeOptions _resizeOptions = new()
    {
        Mode = ResizeMode.Pad,
        PadColor = Color.Black,
        Size = new Size(32, 32),
    };

    private readonly InferenceSession _session;

    /// <summary>
    /// Initializes a new instance of the <see cref="OpenVinoOpenClosedEye0001"/> class.
    /// </summary>
    /// <param name="options">Provide a path to the ONNX model file and customize the behaviour of <see cref="OpenVinoOpenClosedEye0001"/>.</param>
    /// <param name="sessionOptions"><see cref="SessionOptions"/> to customize OnnxRuntime's behaviour.</param>
    public OpenVinoOpenClosedEye0001(OpenVinoOpenClosedEye0001Options options, SessionOptions? sessionOptions = null)
    {
        _ = options?.ModelPath ?? throw new ArgumentException("A model path is required in options.ModelPath.", nameof(options));
        Options = options;
        _session = sessionOptions is null ? new(options.ModelPath) : new(options.ModelPath, sessionOptions);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="OpenVinoOpenClosedEye0001"/> class.
    /// </summary>
    /// <param name="model">OpenVino's open-closed-eye-0001/open_closed_eye.onnx model with 1x3x32x32 BGR input.</param>
    /// <param name="options">Options to customize the behaviour of <see cref="ArcFaceEmbeddingsGenerator"/>. If options.ModelPath is set, it is ignored. The model provided in <paramref name="model"/> takes precedence.</param>
    /// <param name="sessionOptions"><see cref="SessionOptions"/> to customize OnnxRuntime's behaviour.</param>
    public OpenVinoOpenClosedEye0001(byte[] model, OpenVinoOpenClosedEye0001Options? options = null, SessionOptions? sessionOptions = null)
    {
        _ = model ?? throw new ArgumentNullException(nameof(model));
        Options = options ?? new();
        _session = sessionOptions is null ? new(model) : new(model, sessionOptions);
    }

    public OpenVinoOpenClosedEye0001Options Options { get; }

    public void Dispose() => _session.Dispose();

    public bool IsOpen(Image<Rgb24> eyeImage)
    {
        eyeImage.EnsureProperlySizedDestructive(_resizeOptions, !Options.AutoResizeInputToModelDimensions);

        var input = CreateImageTensor(eyeImage);

        var inputMeta = _session.InputMetadata;
        var name = inputMeta.Keys.First();

        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(name, input) };
        using var outputs = _session.Run(inputs);
        var firstOut = outputs.First();
        var tens = firstOut.Value as DenseTensor<float> ?? firstOut.AsTensor<float>().ToDenseTensor();
        Debug.Assert(tens.Length % 2 == 0, "Output tensor length is invalid.");

        var span = tens.Buffer.Span;
        return span[0] < span[1];
    }

    internal static DenseTensor<float> CreateImageTensor(Image<Rgb24> img)
    {
        // The model uses the bgr values, the ints converted to float, no further preprocessing needed.
        var mean = new[] { 0.5f, 0.5f, 0.5f };
        var stdDevVal = 1f;
        var stdDev = new[] { stdDevVal, stdDevVal, stdDevVal };
        var inputDim = new[] { 1, 3, 32, 32 };
        return img.ToTensor(mean, stdDev, inputDim, true);
    }
}

public record OpenVinoOpenClosedEye0001Options
{
    /// <summary>
    /// Gets the path to the onnx file that contains open-closed-eye-0001/open_closed_eye.onnx with 1x3x32x32 BGR input.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Resize the image to dimensions supported by the model if required. This detector throws an
    /// exception if this is set to false and an image is passed in unsupported dimensions.
    /// </summary>
    public bool AutoResizeInputToModelDimensions { get; set; } = true;
}
