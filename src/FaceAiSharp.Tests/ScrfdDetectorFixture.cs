// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Reflection;
using Microsoft.ML.OnnxRuntime;

namespace FaceAiSharp.Tests;

public sealed class ScrfdDetectorFixture : IDisposable
{
    public ScrfdDetectorFixture()
    {
        Detector = CreateScrfdDetector();
    }

    public ScrfdDetector Detector { get; }

    public void Dispose() => Detector.Dispose();

    private static ScrfdDetector CreateScrfdDetector(SessionOptions? sessionOptions = null)
    {
        var modelPath = Path.Combine(GetExeDir(), "onnx", "scrfd_2.5g_kps.onnx");
        var opt = new ScrfdDetectorOptions() { ModelPath = modelPath };
        return new ScrfdDetector(opt, sessionOptions);
    }

    private static string GetExeDir() => Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!;
}
