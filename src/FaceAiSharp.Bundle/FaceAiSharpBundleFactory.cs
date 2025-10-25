// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Reflection;
using Microsoft.ML.OnnxRuntime;

namespace FaceAiSharp;

public static class FaceAiSharpBundleFactory
{
    public static IFaceDetector CreateFaceDetector(SessionOptions? sessionOptions = null) => CreateFaceDetectorWithLandmarks(sessionOptions);

    public static IFaceLandmarksDetector CreateFaceLandmarksDetector(SessionOptions? sessionOptions = null) => CreateFaceDetectorWithLandmarks(sessionOptions);

    public static IFaceDetectorWithLandmarks CreateFaceDetectorWithLandmarks(SessionOptions? sessionOptions = null)
    {
        var modelPath = Path.Combine(GetExeDir(), "onnx", "scrfd_2.5g_kps.onnx");
        var opt = new ScrfdDetectorOptions() { ModelPath = modelPath };
        return new ScrfdDetector(opt, sessionOptions);
    }

    public static IFaceEmbeddingsGenerator CreateFaceEmbeddingsGenerator(SessionOptions? sessionOptions = null)
    {
        var modelPath = Path.Combine(GetExeDir(), "onnx", "arcfaceresnet100-11-int8.onnx");
        var opt = new ArcFaceEmbeddingsGeneratorOptions() { ModelPath = modelPath };
        return new ArcFaceEmbeddingsGenerator(opt, sessionOptions);
    }

    public static IEyeStateDetector CreateEyeStateDetector(SessionOptions? sessionOptions = null)
    {
        var modelPath = Path.Combine(GetExeDir(), "onnx", "open_closed_eye.onnx");
        var opt = new OpenVinoOpenClosedEye0001Options() { ModelPath = modelPath };
        return new OpenVinoOpenClosedEye0001(opt, sessionOptions);
    }

    private static string GetExeDir() => Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!;
}
